"""AdamW with augmented-Lagrangian constraints, against problems with known optima."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.optimizers.adamw import AdamW
from tests.test_mma import CANTILEVER_OPTIMUM, _cantilever


def test_cantilever_benchmark():
    """Svanberg's cantilever again: first-order, so looser than MMA but same optimum."""
    x = np.full(5, 5.0)
    opt = AdamW(n=5, m=1, xmin=1e-3, xmax=10.0, lr=0.05, lr_decay=0.998, lr_min=1e-5, rho=1.0)
    for _ in range(4000):
        x = opt.update(x, *_cantilever(x))

    f0, _, fi, _ = _cantilever(x)
    assert f0 == pytest.approx(1.3399, abs=2e-3)
    assert np.max(np.abs(x - CANTILEVER_OPTIMUM)) < 1e-2
    assert fi[0] <= 2e-3


def test_separable_quadratic_with_active_constraint():
    """min ||x||^2 s.t. sum(x) >= 1 gives x_j = 1/n, with the constraint active."""
    n = 6
    x = np.full(n, 0.9)
    opt = AdamW(n=n, m=1, xmin=0.0, xmax=2.0, lr=0.02, lr_decay=0.998, lr_min=1e-5, rho=100.0)
    for _ in range(4000):
        x = opt.update(x, x @ x, 2.0 * x, np.array([1.0 - x.sum()]), -np.ones((1, n)))
    assert np.allclose(x, 1.0 / n, atol=1e-3)
    assert opt.multipliers[0] > 0.0, "an active constraint must carry a positive multiplier"


def test_inactive_constraint_keeps_a_zero_multiplier():
    """A constraint that never binds must not perturb the unconstrained optimum."""
    n = 4
    target = np.array([0.2, 0.4, 0.6, 0.8])
    x = np.full(n, 0.5)
    opt = AdamW(n=n, m=1, lr=0.02, lr_decay=0.998, lr_min=1e-5)
    for _ in range(3000):
        f0 = float(((x - target) ** 2).sum())
        x = opt.update(x, f0, 2.0 * (x - target), np.array([x.sum() - 10.0]), np.ones((1, n)))
    assert np.allclose(x, target, atol=1e-3)
    assert opt.multipliers[0] == 0.0


def test_bounds_hold_and_steps_are_capped_by_the_rate():
    """Clipping keeps every iterate in the box; Adam's step is at most ~lr per variable."""
    n = 20
    rng = np.random.default_rng(0)
    target = rng.uniform(-0.5, 1.5, n)  # partly outside the box
    x = np.full(n, 0.5)
    opt = AdamW(n=n, m=1, lr=0.05)
    for _ in range(200):
        f0 = float(((x - target) ** 2).sum())
        xnew = opt.update(
            x, f0, 2.0 * (x - target), np.array([x.mean() - 0.9]), np.full((1, n), 1.0 / n)
        )
        assert np.all((xnew >= 0.0) & (xnew <= 1.0))
        # Bias-corrected Adam steps are bounded by roughly lr / (1 - beta1).
        assert np.max(np.abs(xnew - x)) <= opt.lr * 10.0
        x = xnew
    # Variables whose target lies outside the box end up on the bound.
    assert np.all(x[target < 0.0] < 0.05)


def test_learning_rate_decays_to_its_floor():
    opt = AdamW(n=1, m=0, lr=0.02, lr_decay=0.5, lr_min=0.004)
    rates = []
    for _ in range(6):
        rates.append(opt.current_lr)
        opt.update(np.array([0.5]), 0.0, np.array([1.0]), np.zeros(0), np.zeros((0, 1)))
    assert rates[:3] == pytest.approx([0.02, 0.01, 0.005])
    assert rates[-1] == pytest.approx(0.004)


def test_weight_decay_pulls_towards_its_target():
    """With a flat objective, decoupled decay alone moves x to decay_target."""
    x = np.array([0.1, 0.9])
    opt = AdamW(n=2, m=0, lr=0.05, lr_decay=1.0, weight_decay=1.0, decay_target=0.5)
    for _ in range(400):
        x = opt.update(x, 0.0, np.zeros(2), np.zeros(0), np.zeros((0, 2)))
    assert np.allclose(x, 0.5, atol=1e-3)
