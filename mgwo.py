import numpy as np

from constants import EPSILON, GWO_TOP_K


class MGWOptimizer:
    """Modified Gray Wolf Optimizer for particle refinement.

    Each call to :meth:`optimize` runs ``mgwo_max_iter`` GWO iterations.
    The three best-weighted particles are treated as the alpha, beta, and
    delta wolves; all other particles are attracted toward them.  A particle
    is only replaced when the candidate position scores a higher likelihood
    than the current position.

    Parameters
    ----------
    mgwo_max_iter:
        Number of GWO iterations to run per :meth:`optimize` call.
    a:
        Initial value of the GWO control parameter *a* (typically 2).
    rng:
        NumPy ``Generator`` instance used for reproducible randomness.
    lh_eval:
        Likelihood evaluator used to score candidate particle positions.
    """

    def __init__(self, mgwo_max_iter: int, a: float, rng: np.random.Generator, lh_eval) -> None:
        self.lh_eval = lh_eval
        self.mgwo_max_iter = mgwo_max_iter
        self.a = a
        self.rng = rng

    def optimize(
        self,
        frame: np.ndarray,
        particles: np.ndarray,
        weights: np.ndarray,
        std_mgwo: float,
    ) -> None:
        """Refine *particles* and *weights* in-place using the GWO update rule.

        Parameters
        ----------
        frame:
            Current video frame (BGR).
        particles:
            Array of shape ``(N, dim)`` representing the current particle set;
            updated in-place.
        weights:
            Array of shape ``(N,)`` with current particle weights; updated
            in-place.
        std_mgwo:
            Standard deviation applied to the random GWO step coefficients
            ``r1`` and ``r2``.
        """
        N = particles.shape[0]
        dim = particles.shape[1]

        self.a = 2
        particles_new = np.empty((N, dim))
        weights_new = np.empty((N, 1))

        for t in range(self.mgwo_max_iter):
            r1 = 0.5 + self.rng.standard_normal((N, GWO_TOP_K, dim)) * std_mgwo
            r2 = 0.5 + self.rng.standard_normal((N, GWO_TOP_K, dim)) * std_mgwo
            A = 2 * self.a * r1 - self.a
            C = 2 * r2

            # Select the GWO_TOP_K best particles as alpha, beta, delta wolves
            ind = np.argpartition(weights, -GWO_TOP_K)[-GWO_TOP_K:]
            ind = ind[np.argsort(weights[ind])]

            alpha = ind[2]
            beta = ind[1]
            delta = ind[0]

            X_alpha = particles[alpha]
            X_beta = particles[beta]
            X_delta = particles[delta]

            # Attraction step
            D_alpha = np.abs(C[:, 0, :] * X_alpha - particles)
            D_beta = np.abs(C[:, 1, :] * X_beta - particles)
            D_delta = np.abs(C[:, 2, :] * X_delta - particles)

            X_1 = X_alpha - A[:, 0, :] * D_alpha
            X_2 = X_beta - A[:, 1, :] * D_beta
            X_3 = X_delta - A[:, 2, :] * D_delta

            particles_new = (X_1 + X_2 + X_3) / 3

            # Accept candidate positions that improve likelihood
            self.lh_eval.update(frame, particles, weights, normalize=False)
            self.lh_eval.update(frame, particles_new, weights_new, normalize=False)
            for i, (weight, weight_new) in enumerate(zip(weights, weights_new)):
                if weight_new > weight:
                    particles[i] = particles_new[i]
                    weights[i] = weights_new[i]

            # Decay control parameter (cosine annealing from 2 → 0)
            phi = np.pi * t / self.mgwo_max_iter
            self.a = 2 - 2 * (np.sin(phi / 2)) ** 2

        weights /= np.sum(weights) + EPSILON
