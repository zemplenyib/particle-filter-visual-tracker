import logging
import os

import cv2
import numpy as np

from constants import EPSILON, HIST_BINS, HIST_RANGES
from display_images import extract_rotated_roi, get_rectangle

logger = logging.getLogger(__name__)


def load_data(path: str) -> tuple[list, list, np.ndarray]:
    """Load images and ground-truth bounding boxes from a dataset directory.

    The directory must contain:
    - ``img/``  — image files in sorted filename order
    - ``groundtruth_rect.txt``  — one bounding box per line, comma- or
      space-separated as ``x y w h``

    Parameters
    ----------
    path:
        Root directory of the dataset.

    Returns
    -------
    images:
        List of BGR images as ``np.ndarray``.
    filenames:
        Corresponding list of image filenames.
    gt:
        Ground-truth array of shape ``(N, 4)`` with columns ``[x, y, w, h]``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base path not found: {path}")

    images: list[np.ndarray] = []
    filenames: list[str] = []

    img_path = os.path.join(path, "img")
    if not os.path.isdir(img_path):
        logger.warning("Image directory missing at %s", img_path)
    for filename in sorted(os.listdir(img_path)):
        img = cv2.imread(os.path.join(img_path, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)

    gt_path = os.path.join(path, "groundtruth_rect.txt")
    with open(gt_path, "r") as f:
        # Handles both comma- and space-separated files.
        gt = [
            [int(float(x)) for x in line.replace(",", " ").split()]
            for line in f
        ]
    gt_arr = np.array(gt)

    return images, filenames, gt_arr


def iou(bb1: dict, bb2: dict) -> float:
    """Return the Intersection over Union (IoU) of two axis-aligned bounding boxes.

    Parameters
    ----------
    bb1, bb2:
        Dicts with keys ``'x1'``, ``'x2'``, ``'y1'``, ``'y2'`` where
        ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the
        bottom-right corner.

    Returns
    -------
    float
        IoU value in ``[0, 1]``.
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    bb1_area = (bb1["x2"] - bb1["x1"] + 1) * (bb1["y2"] - bb1["y1"] + 1)
    bb2_area = (bb2["x2"] - bb2["x1"] + 1) * (bb2["y2"] - bb2["y1"] + 1)

    result = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert 0.0 <= result <= 1.0
    return result


class LikelihoodEvaluator:
    """Evaluates particle likelihoods via histogram-based appearance matching.

    A reference colour histogram is computed from the target region in the
    first frame.  Each subsequent call to :meth:`compute_weight` measures the
    Bhattacharyya similarity between that reference and the histogram of the
    candidate region.

    Parameters
    ----------
    w_init, h_init:
        Width and height of the target bounding box in the first frame.
    initial_frame:
        First video frame (BGR, ``np.ndarray``).
    state:
        Initial state vector ``[x, vx, y, vy, theta, scale]``.
    """

    def __init__(
        self,
        w_init: float,
        h_init: float,
        initial_frame: np.ndarray,
        state: np.ndarray,
    ) -> None:
        self.w_init = w_init
        self.h_init = h_init
        roi = extract_rotated_roi(
            initial_frame,
            (state[0], state[2]),
            state[5] * self.w_init,
            state[5] * self.h_init,
            state[4],
        )
        self.target_hist = self.get_histogram_from_roi(roi)

    def get_histogram(self, frame: np.ndarray, particle: np.ndarray) -> np.ndarray:
        """Return a normalised BGR histogram for the axis-aligned ROI of *particle*.

        Parameters
        ----------
        frame:
            Current video frame.
        particle:
            State vector, or ``None`` to use the full frame.

        Returns
        -------
        np.ndarray
            Flattened, normalised histogram.
        """
        if particle is not None:
            x1, x2, y1, y2 = get_rectangle(particle, self.w_init, self.h_init)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
        return self.get_histogram_from_roi(roi)

    def get_histogram_from_roi(self, roi: np.ndarray) -> np.ndarray:
        """Return a normalised BGR histogram for an already-cropped ROI.

        Parameters
        ----------
        roi:
            Cropped image region (BGR, ``np.ndarray``).

        Returns
        -------
        np.ndarray
            Flattened, normalised histogram of length ``prod(HIST_BINS)``.
        """
        hist = cv2.calcHist([roi], [0, 1, 2], None, HIST_BINS, HIST_RANGES)
        hist = hist.flatten() / (np.sum(hist.flatten()) + EPSILON)
        return hist

    def get_histogram_using_mask(self, frame: np.ndarray, particle: np.ndarray) -> np.ndarray:
        """Return a normalised histogram computed inside the rotated ROI mask.

        Uses a polygon mask to exclude pixels outside the rotated bounding
        box, giving a more accurate appearance descriptor than the
        axis-aligned crop used by :meth:`get_histogram`.

        Parameters
        ----------
        frame:
            Current video frame.
        particle:
            State vector ``[x, vx, y, vy, theta, scale]``.

        Returns
        -------
        np.ndarray
            Flattened, normalised histogram of length ``prod(HIST_BINS)``.
        """
        center = (particle[0], particle[2])
        width = particle[5] * self.w_init
        height = particle[5] * self.h_init
        theta = particle[4]
        rect = ((center[0], center[1]), (width, height), theta)
        box = cv2.boxPoints(rect).astype(int)

        x_min, y_min = box[:, 0].min(), box[:, 1].min()
        x_max, y_max = box[:, 0].max(), box[:, 1].max()

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return np.zeros(HIST_BINS[0] * HIST_BINS[1] * HIST_BINS[2], dtype=np.float32)

        box_shifted = np.round(box - [x_min, y_min]).astype(np.int32)
        box_shifted = np.ascontiguousarray(box_shifted.reshape((-1, 1, 2)))

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [box_shifted], 255)

        hist = cv2.calcHist([roi], [0, 1, 2], mask, HIST_BINS, HIST_RANGES)
        hist = hist.flatten() / (np.sum(hist.flatten()) + EPSILON)
        return hist

    def compute_weight(self, frame: np.ndarray, particle: np.ndarray) -> float:
        """Return the appearance similarity weight for a single particle.

        Parameters
        ----------
        frame:
            Current video frame.
        particle:
            State vector ``[x, vx, y, vy, theta, scale]``.

        Returns
        -------
        float
            Bhattacharyya-based similarity in ``[0, 1]`` (higher = more similar).
        """
        particle_hist = self.get_histogram_using_mask(frame, particle)
        weight = 1 - cv2.compareHist(
            self.target_hist, particle_hist, cv2.HISTCMP_BHATTACHARYYA
        )
        return weight

    def update(
        self,
        frame: np.ndarray,
        particles: np.ndarray,
        weights: np.ndarray,
        normalize: bool = True,
    ) -> None:
        """Update *weights* in-place for every particle in *particles*.

        Parameters
        ----------
        frame:
            Current video frame.
        particles:
            Array of shape ``(N, dim)``.
        weights:
            Array of shape ``(N,)`` or ``(N, 1)``; updated in-place.
        normalize:
            If ``True`` (default), weights are divided by their sum so they
            form a proper probability distribution.
        """
        for i, particle in enumerate(particles):
            weights[i] = self.compute_weight(frame, particle)

        if normalize:
            weights /= np.sum(weights) + EPSILON
