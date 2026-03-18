# Particle Filter with Modified Gray Wolf Optimizer

A Python implementation of a visual object tracker that combines a **Particle Filter** (Bayesian state estimator) with a **Modified Gray Wolf Optimizer** (swarm-based particle refinement) for robust single-object tracking in video sequences.

---

## Algorithm Overview

### Particle Filter — Bayesian State Estimation

The tracker maintains *N* particles, each representing a candidate object state:

```
state = [x, vx, y, vy, θ, s]
         ^   ^   ^   ^  ^  ^
         |   |   |   |  |  scale
         |   |   |   |  rotation (deg)
         |   |   |   vy (px/frame)
         |   |   y center (px)
         |   vx (px/frame)
         x center (px)
```

Each frame executes the following cycle:

```
┌──────────┐      ┌────────────┐      ┌────────────┐      ┌──────────┐     ┌──────────┐
│  Predict │─── ▶│   Weight    │────▶│  MGWO      │────▶│ Estimate │────▶│ Resample │
│ (motion) │      │(appearance)│      │ (refine)   │      │ (mean)   │     │(sys res) │
└──────────┘      └────────────┘      └────────────┘      └──────────┘     └──────────┘
```

1. **Predict** — propagate each particle through a constant-velocity motion model with Gaussian process noise.
2. **Weight** — score each particle by comparing its colour histogram (computed inside the rotated bounding box) to the reference target histogram using the Bhattacharyya distance.
3. **MGWO refine** — pull particles toward high-likelihood regions (see below).
4. **Estimate** — compute the weighted mean across all particles.
5. **Resample** — systematic resampling to focus particles on high-weight regions.

### Modified Gray Wolf Optimizer — Guided Particle Refinement

After weighting, the three highest-weight particles are designated as the **α**, **β**, and **δ** wolves.  Every other particle is attracted toward them using the GWO update rule:

```
D_α = |C₁·X_α − X|,   X₁ = X_α − A₁·D_α
D_β = |C₂·X_β − X|,   X₂ = X_β − A₂·D_β
D_δ = |C₃·X_δ − X|,   X₃ = X_δ − A₃·D_δ

X_new = (X₁ + X₂ + X₃) / 3
```

A particle is only replaced by its candidate position when the candidate scores a **higher likelihood**, making the step a guided hill-climb rather than a blind attraction.  The control parameter *a* decays from 2 → 0 across iterations (cosine schedule), transitioning from exploration to exploitation.

---

## Results

Average IoU across 10 standard OTB benchmark sequences (100 particles, 10 GWO iterations):

| Dataset   | Without MGWO | MGWO (1 iter) | MGWO (10 iter) |
|-----------|:------------:|:-------------:|:--------------:|
| BlurBody  | 0.309        | 0.452         | **0.481**      |
| BlurOwl   | 0.263        | 0.502         | **0.497**      |
| Box       | 0.164        | **0.231**     | 0.140          |
| Dog       | 0.367        | 0.421         | **0.469**      |
| Girl      | 0.331        | 0.404         | **0.406**      |
| Jogging1  | 0.440        | 0.306         | **0.483**      |
| Jogging2  | 0.503        | 0.340         | **0.532**      |
| Skating1  | 0.403        | **0.414**     | 0.382          |
| Skating2  | 0.386        | 0.375         | 0.369          |
| Surfer    | 0.053        | 0.134         | **0.179**      |

MGWO (10 iterations) improves or matches the baseline on 7 of 10 sequences. Overall, the absolute IoU scores leave room for improvement — parameter tuning (particle count, process-noise sigmas, number of GWO iterations) on a per-sequence basis would likely push results significantly higher.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Dataset note:** All 10 benchmark sequences are included in this repository under `Datasets/`.
> Clone size is approximately **1 GB**, so the initial clone may take a while depending on your connection:
> ```bash
> git clone https://github.com/zemplenyib/particle-filter-visual-tracker
> ```
> The full OTB-100 dataset (100 sequences) is available via [prosti221/OTB-dataset](https://github.com/prosti221/OTB-dataset).

---

## Usage

```bash
python main.py --dataset BlurBody
```

**Available datasets** (must be present in `./Datasets/`):

`BlurBody`, `BlurOwl`, `Box`, `Dog`, `Girl`, `Jogging1`, `Jogging2`, `Skating1`, `Skating2`, `Surfer`

Full help:

```bash
python main.py --help
```

---

## Project Structure

```
PF-MGWO/
├── main.py                   # Entry point — argparse, tracker configuration
├── particle_filter_tracker.py # ParticleFilterTracker class (predict/weight/estimate/resample)
├── mgwo.py                   # MGWOptimizer — GWO-based particle refinement
├── utils.py                  # Data loading, IoU metric, LikelihoodEvaluator
├── display_images.py         # Visu class + OpenCV / Matplotlib drawing helpers
├── constants.py              # Shared numerical constants (bins, thresholds, seeds)
├── requirements.txt          # pip dependencies
└── tests/
    └── test_utils.py         # Unit tests (pytest)
```

---

## Key Concepts — Sensor Fusion Perspective

| Component | Role |
|-----------|------|
| **Particle filter** | Sequential Monte Carlo Bayesian estimator; represents the full posterior `p(state | observations)` as a weighted sample set. |
| **Colour histogram likelihood** | Observation model `p(z | state)` — measures how well a candidate region matches the reference appearance. |
| **Constant-velocity motion model** | Transition prior `p(state_t | state_{t-1})` — structured Gaussian noise via a 6×6 process-noise covariance matrix. |
| **GWO refinement** | Guided search within the particle cloud; exploits the likelihood landscape computed during the weight step without additional image evaluations in new locations. |
| **Systematic resampling** | Reduces particle degeneracy after the weight step; preserves diversity while focusing computation on high-probability regions. |

---

## Running the Tests

```bash
python -m pytest tests/ -v
```
