"""Shared constants for the PF-MGWO tracker."""

# Reproducibility
SEED = 42

# Numerical stability
EPSILON = 1e-6

# Colour histogram parameters (8×8×8 bins over BGR [0,256))
HIST_BINS = [8, 8, 8]
HIST_RANGES = [0, 256, 0, 256, 0, 256]

# Process-noise standard deviations for the motion model and GWO step
#   [std_x, std_y, std_theta, std_scale, std_mgwo]
SIGMA = [1.73, 1.5, 0.2, 0.04, 0.0458]

# Particle re-initialisation trigger (weight below this → object likely lost)
REINIT_WEIGHT_THRESHOLD = 0.5

# Gaussian spread used when scattering particles around a poor estimate
#   [std_x, std_vx, std_y, std_vy, std_theta, std_scale]
REINIT_SIGMA = [5, 2, 5, 2, 0.17, 0.5]

# Number of elite (alpha/beta/delta) wolves kept by GWO
GWO_TOP_K = 3
