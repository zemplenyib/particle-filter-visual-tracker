import argparse
import os

from constants import SIGMA
from particle_filter_tracker import ParticleFilterTracker

base_path = os.path.dirname(os.path.abspath(__file__))

DATASETS = [
    "BlurBody",
    "BlurOwl",
    "Box",
    "Dog",
    "Girl",
    "Jogging1",
    "Jogging2",
    "Skating1",
    "Skating2",
    "Surfer",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="particle_filter",
        description=(
            "Visual object tracker using a Particle Filter enhanced by a "
            "Modified Gray Wolf Optimizer (MGWO). "
            "The particle filter provides a Bayesian estimate of the target state; "
            "MGWO refines the particle set each frame by attracting particles toward "
            "the highest-likelihood regions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="BlurBody",
        choices=DATASETS,
        help="Name of the benchmark dataset to track (must exist under ./Datasets/).",
    )
    args = parser.parse_args()

    pfTracker = ParticleFilterTracker(
        N=100,
        sigma=SIGMA,
        velocity=[1, 1],
        T=1,
        use_mgwo=True,
        mgwo_max_iter=10,
    )

    data_path = os.path.join(base_path, "Datasets", args.dataset)
    pfTracker.process_dataset(data_path, visualize=True)
