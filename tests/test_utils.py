"""Unit tests for utils.py — IoU metric and LikelihoodEvaluator."""
import sys
import os

import numpy as np
import pytest

# Allow imports from the project root when running via pytest from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import iou, LikelihoodEvaluator


# ---------------------------------------------------------------------------
# iou()
# ---------------------------------------------------------------------------

def _box(x1, y1, x2, y2):
    """Helper: build the dict format expected by iou()."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_iou_perfect_overlap():
    """Identical boxes must give IoU = 1.0."""
    box = _box(0, 0, 10, 10)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap():
    """Non-overlapping boxes must give IoU = 0.0."""
    bb1 = _box(0, 0, 5, 5)
    bb2 = _box(10, 10, 20, 20)
    assert iou(bb1, bb2) == pytest.approx(0.0)


def test_iou_partial_overlap():
    """Half-overlapping boxes of equal size should give IoU = 1/3.

    bb1 covers x=[0,10], bb2 covers x=[5,15] (y identical for both).
    Intersection width = 6 (pixels 5-10 inclusive), union = 11+11-6 = 16.
    With the +1 pixel-inclusive convention used in utils.iou:
      intersection = 6 * 11 = 66
      bb1_area = bb2_area = 11 * 11 = 121
      IoU = 66 / (121 + 121 - 66) = 66 / 176 = 3/8
    """
    bb1 = _box(0, 0, 10, 10)
    bb2 = _box(5, 0, 15, 10)
    expected = 66 / (121 + 121 - 66)
    assert iou(bb1, bb2) == pytest.approx(expected)


def test_iou_contained_box():
    """A box fully inside another must give IoU equal to the area ratio."""
    outer = _box(0, 0, 10, 10)
    inner = _box(2, 2, 8, 8)
    # inner area = 7*7 = 49, outer area = 11*11 = 121
    # intersection = 49, union = 121
    expected = 49 / 121
    assert iou(outer, inner) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LikelihoodEvaluator
# ---------------------------------------------------------------------------

def _make_evaluator(width=40, height=30):
    """Return a LikelihoodEvaluator initialised on a random BGR frame."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    # state: [x, vx, y, vy, theta, scale]
    state = np.array([100.0, 0.0, 100.0, 0.0, 0.0, 1.0])
    return LikelihoodEvaluator(width, height, frame, state), frame


def test_likelihood_evaluator_weight_shape():
    """update() must fill *weights* for every particle without changing its length."""
    evaluator, frame = _make_evaluator()
    N = 20
    particles = np.tile([100.0, 0.0, 100.0, 0.0, 0.0, 1.0], (N, 1))
    weights = np.ones(N) / N
    evaluator.update(frame, particles, weights)
    assert weights.shape == (N,), "weights array length must not change"


def test_likelihood_evaluator_weights_sum_to_one():
    """After a normalised update, weights must sum to 1."""
    evaluator, frame = _make_evaluator()
    N = 15
    particles = np.tile([100.0, 0.0, 100.0, 0.0, 0.0, 1.0], (N, 1))
    weights = np.ones(N) / N
    evaluator.update(frame, particles, weights, normalize=True)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_likelihood_evaluator_target_weight_high():
    """A particle centred exactly on the target should score higher than a random one."""
    evaluator, frame = _make_evaluator()
    target_particle = np.array([100.0, 0.0, 100.0, 0.0, 0.0, 1.0])
    off_particle = np.array([10.0, 0.0, 10.0, 0.0, 0.0, 1.0])
    w_target = evaluator.compute_weight(frame, target_particle)
    w_off = evaluator.compute_weight(frame, off_particle)
    assert w_target >= w_off
