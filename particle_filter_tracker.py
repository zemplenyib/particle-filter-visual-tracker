import logging

import numpy as np
from filterpy.monte_carlo import systematic_resample

import utils
from constants import EPSILON, REINIT_SIGMA, REINIT_WEIGHT_THRESHOLD, SEED
from display_images import Visu, get_rectangle
from mgwo import MGWOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname).3s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ParticleFilterTracker:
    """Particle filter tracker optionally enhanced by a Modified Gray Wolf Optimizer.

    The tracker maintains a set of *N* particles, each representing a
    candidate object state ``[x, vx, y, vy, theta, scale]``.  At every
    frame the standard predict → weight → (MGWO refine) → estimate →
    resample loop is executed.

    Parameters
    ----------
    N:
        Number of particles.
    sigma:
        List ``[std_x, std_y, std_theta, std_scale, std_mgwo]`` of process-
        noise standard deviations (pixel / radian / scale / GWO units).
    velocity:
        Initial velocity ``[vx, vy]`` for the first-frame state vector.
    T:
        Sampling period (frames); used to build the constant-velocity motion
        matrix.
    use_mgwo:
        Whether to apply the GWO refinement step after weighting.
    mgwo_max_iter:
        Number of GWO iterations per frame.
    """

    def __init__(
        self,
        N: int,
        sigma: list[float],
        velocity: list[float],
        T: float,
        use_mgwo: bool,
        mgwo_max_iter: int,
    ) -> None:
        self._N = N
        self._dim = 0
        self._sigma = sigma
        self._velocity = velocity
        self._T = T
        self._use_mgwo = use_mgwo
        self._mgwo_max_iter = mgwo_max_iter
        self._particles: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._w_init = 0
        self._h_init = 0
        self._G: np.ndarray | None = None
        self._Q: np.ndarray | None = None
        self._rng = np.random.default_rng(seed=SEED)
        self._lh_eval: utils.LikelihoodEvaluator | None = None
        self._mgwo: MGWOptimizer | None = None
        self._visu = Visu()

    def initialize(
        self,
        initial_state: np.ndarray,
        w_init: float,
        h_init: float,
        frame: np.ndarray,
    ) -> None:
        """Set up particles, motion model, and appearance model for the first frame.

        Parameters
        ----------
        initial_state:
            State vector ``[x, vx, y, vy, theta, scale]`` for the target
            centroid in the first frame.
        w_init, h_init:
            Width and height of the target bounding box in the first frame.
        frame:
            First video frame (BGR).
        """
        np.random.seed(SEED)
        self._w_init = w_init
        self._h_init = h_init
        self._dim = initial_state.shape[0]

        # Constant-velocity motion matrix
        self._G = np.array(
            [
                [1, self._T, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, self._T, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        sigma_x = self._sigma[0]
        sigma_y = self._sigma[1]
        sigma_theta = self._sigma[2]
        sigma_s = self._sigma[3]

        Qx = np.array(
            [
                [1 / 4 * self._T**4, 1 / 2 * self._T**3],
                [1 / 2 * self._T**3, self._T**2],
            ]
        ) * sigma_x**2
        Qy = np.array(
            [
                [1 / 4 * self._T**4, 1 / 2 * self._T**3],
                [1 / 2 * self._T**3, self._T**2],
            ]
        ) * sigma_y**2

        self._Q = np.zeros((6, 6))
        self._Q[0:2, 0:2] = Qx
        self._Q[2:4, 2:4] = Qy
        self._Q[4, 4] = sigma_theta**2
        self._Q[5, 5] = sigma_s**2

        self._particles = self.create_clone_particles(initial_state)
        self._weights = np.ones(self._N) / self._N

        self._lh_eval = utils.LikelihoodEvaluator(
            self._w_init, self._h_init, frame, initial_state
        )
        self._mgwo = MGWOptimizer(self._mgwo_max_iter, 2, self._rng, self._lh_eval)

    def predict(self) -> None:
        """Propagate particles forward using the constant-velocity motion model."""
        if self._G is not None and self._Q is not None:
            for i, particle in enumerate(self._particles):
                mean = self._G.dot(particle)
                self._particles[i] = self._rng.multivariate_normal(mean, self._Q)
        else:
            self._particles = self._rng.normal(self._particles, self._sigma)

    def estimate(self) -> np.ndarray:
        """Return the weighted-mean state estimate across all particles."""
        return np.sum(self._particles * self._weights[:, None], axis=0)

    def verify(self, state_estimation: np.ndarray, ground_truth: np.ndarray | None) -> float | None:
        """Compute IoU between the estimated bounding box and ground truth.

        Parameters
        ----------
        state_estimation:
            Estimated state vector.
        ground_truth:
            Ground-truth row ``[x, y, w, h]``, or ``None`` to skip.

        Returns
        -------
        float or None
            IoU in ``[0, 1]``, or ``None`` if *ground_truth* is ``None``.
        """
        if ground_truth is None:
            return None

        x1, x2, y1, y2 = get_rectangle(state_estimation, self._w_init, self._h_init)

        if x1 < x2:
            bb_estimation = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        elif x1 > x2:
            bb_estimation = {"x1": x2, "y1": y1, "x2": x1, "y2": y2}
        else:
            bb_estimation = {"x1": x1, "y1": y1, "x2": x2 + 1, "y2": y2}

        bb_gt = {
            "x1": ground_truth[0],
            "y1": ground_truth[1],
            "x2": ground_truth[0] + ground_truth[2],
            "y2": ground_truth[1] + ground_truth[3],
        }

        return utils.iou(bb_gt, bb_estimation)

    def create_clone_particles(self, initial: np.ndarray) -> np.ndarray:
        """Return an ``(N, dim)`` array of identical copies of *initial*."""
        return np.tile(initial, (self._N, 1))

    def create_gaussian_particles(
        self,
        mean: list | np.ndarray,
        std: list | np.ndarray,
        N: int,
        dim: int,
    ) -> np.ndarray:
        """Return *N* particles sampled from an isotropic Gaussian.

        Parameters
        ----------
        mean:
            Centre of the distribution (length *dim*).
        std:
            Per-dimension standard deviations (length *dim*).
        N:
            Number of particles to create.
        dim:
            State-space dimensionality.

        Returns
        -------
        np.ndarray
            Array of shape ``(N, dim)``.
        """
        mean = np.asarray(mean)
        std = np.asarray(std)
        return mean + self._rng.standard_normal((N, dim)) * std

    def resample_from_index(self, indexes: np.ndarray) -> None:
        """Replace particles with the subset selected by *indexes* and reset weights."""
        self._particles[:] = self._particles[indexes]
        self._weights.resize(len(self._particles))
        self._weights.fill(1.0 / len(self._weights))

    def resample(self) -> None:
        """Perform systematic resampling based on current particle weights."""
        indexes = systematic_resample(self._weights)
        self.resample_from_index(indexes)
        assert np.allclose(self._weights, 1 / self._N)

    def reinit_if_needed(self, frame: np.ndarray, estimation: np.ndarray) -> bool:
        """Re-scatter particles around *estimation* if tracking confidence is low.

        Parameters
        ----------
        frame:
            Current video frame.
        estimation:
            Current state estimate.

        Returns
        -------
        bool
            ``True`` if particles were re-initialised, ``False`` otherwise.
        """
        weight = self._lh_eval.compute_weight(frame, estimation)
        if weight < REINIT_WEIGHT_THRESHOLD:
            self._particles = self.create_gaussian_particles(
                estimation, REINIT_SIGMA, self._N, self._dim
            )
            return True
        return False

    def apply_mgwo_optimizer(self, frame: np.ndarray) -> None:
        """Run the GWO refinement step; re-scatter particles if they all have zero weight.

        Parameters
        ----------
        frame:
            Current video frame.
        """
        if np.sum(self._weights) <= EPSILON:
            height, width, _ = frame.shape
            self._particles = self.create_gaussian_particles(
                [int(width / 2), 1, int(height / 2), 1, 1, 1],
                [int(width / 3), 2, int(height / 3), 2, 0.17, 0.5],
                self._N,
                self._dim,
            )
            self._lh_eval.update(frame, self._particles, self._weights)
            logger.info("Object lost")

        self._mgwo.optimize(frame, self._particles, self._weights, self._sigma[4])

    def process_dataset(self, dataset: str, visualize: bool = False) -> None:
        """Run the tracker on every frame of a dataset directory.

        Parameters
        ----------
        dataset:
            Path to the dataset root (must contain ``img/`` and
            ``groundtruth_rect.txt``).
        visualize:
            If ``True``, display each frame in an OpenCV window.
        """
        images, _, gt = utils.load_data(dataset)
        initial_state = np.array(
            [
                gt[0, 0] + gt[0, 2] / 2,
                self._velocity[0],
                gt[0, 1] + gt[0, 3] / 2,
                self._velocity[1],
                0,
                1,
            ]
        )

        self.initialize(
            initial_state=initial_state,
            w_init=gt[0, 2],
            h_init=gt[0, 3],
            frame=images[0],
        )

        state_estimate_arr: list[np.ndarray] = []
        iou_arr: list[float] = []

        for index, frame_bgr in enumerate(images):
            if frame_bgr is not None:
                logger.info("Processing frame: %d / %d", index, len(images))
                state_estimation, iou = self.step(frame_bgr, gt[index, :])

                iou_arr.append(iou)
                state_estimate_arr.append(state_estimation)

                if visualize:
                    self._visu.draw_frame_cv2(
                        frame_bgr,
                        self._particles,
                        state_estimation,
                        index,
                        self._w_init,
                        self._h_init,
                        iou,
                    )
                    self._visu.show()

        iou_avg = sum(iou_arr) / len(iou_arr)
        logger.info("IoU avg = %.3f", iou_avg)

    def step(
        self, frame: np.ndarray, gt: np.ndarray | None = None
    ) -> tuple[np.ndarray, float | None]:
        """Execute one full predict–weight–refine–estimate–resample cycle.

        Parameters
        ----------
        frame:
            Current video frame (BGR).
        gt:
            Ground-truth bounding box ``[x, y, w, h]`` for this frame, or
            ``None`` if unavailable.

        Returns
        -------
        estimation:
            Estimated state vector for this frame.
        iou:
            IoU with ground truth, or ``None`` if *gt* is ``None``.
        """
        # 1. Propagate particles through the motion model
        self.predict()

        # 2. Compute appearance-based weights
        self._lh_eval.update(frame, self._particles, self._weights)

        # 3. Refine particles with Modified Gray Wolf Optimizer
        if self._use_mgwo:
            self.apply_mgwo_optimizer(frame)

        # 4. Weighted-mean state estimate
        estimation = self.estimate()

        # 5. Resample according to weights
        self.resample()

        # Re-initialise if the tracker has drifted too far from the target
        if self.reinit_if_needed(frame, estimation):
            return estimation, 0.0

        # 6. Evaluate against ground truth
        iou = self.verify(estimation, gt)

        return estimation, iou
