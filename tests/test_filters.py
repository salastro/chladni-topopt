"""Density filter and threshold projection."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.filters import (
    BetaContinuation,
    DensityFilter,
    ExponentialThreshold,
)


def test_filter_preserves_a_uniform_field(mesh):
    """Row-normalisation means a constant design filters to itself."""
    f = DensityFilter(rmin=3.0 * mesh.dx, centroids=mesh.elem_centroids())
    x = np.full(mesh.n_elem, 0.42)
    assert np.allclose(f.forward(x), 0.42)
    assert np.allclose(np.asarray(f.matrix.sum(axis=1)).ravel(), 1.0)


def test_filter_smooths_and_is_self_adjoint_in_the_right_direction(mesh):
    """``backward`` must apply ``H^T``, not ``H``."""
    f = DensityFilter(rmin=3.0 * mesh.dx, centroids=mesh.elem_centroids())
    rng = np.random.default_rng(1)
    u = rng.normal(size=mesh.n_elem)
    v = rng.normal(size=mesh.n_elem)
    # <H u, v> == <u, H^T v>
    assert f.forward(u) @ v == pytest.approx(u @ f.backward(v), rel=1e-12)


def test_filter_radius_controls_support(mesh):
    narrow = DensityFilter(rmin=1.1 * mesh.dx, centroids=mesh.elem_centroids())
    wide = DensityFilter(rmin=4.0 * mesh.dx, centroids=mesh.elem_centroids())
    assert narrow.matrix.nnz < wide.matrix.nnz


@pytest.mark.parametrize("beta", [1.0, 8.0, 64.0, 256.0])
def test_projection_fixes_the_unit_interval_and_the_threshold(beta):
    p = ExponentialThreshold(eta=0.5, beta=beta)
    assert p.forward(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert p.forward(np.array([0.5]))[0] == pytest.approx(0.5, abs=1e-12)
    assert p.forward(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-12)


def test_projection_degenerates_to_identity_at_zero_beta():
    p = ExponentialThreshold(eta=0.5, beta=1e-9)
    x = np.linspace(0.0, 1.0, 21)
    assert np.allclose(p.forward(x), x, atol=1e-8)


@pytest.mark.parametrize("beta", [1.0, 8.0, 64.0, 256.0])
def test_projection_is_monotone_and_sharpens_with_beta(beta):
    p = ExponentialThreshold(eta=0.5, beta=beta)
    x = np.linspace(0.0, 1.0, 201)
    y = p.forward(x)
    assert np.all(np.diff(y) >= -1e-12)
    assert np.all((y >= -1e-12) & (y <= 1.0 + 1e-12))
    # Larger beta pushes values further from the threshold.
    mild = ExponentialThreshold(eta=0.5, beta=1.0).forward(x)
    assert np.sum(np.abs(y - 0.5)) >= np.sum(np.abs(mild - 0.5)) - 1e-12


@pytest.mark.parametrize("beta", [1.0, 8.0, 64.0, 256.0])
def test_projection_gradient_matches_finite_differences(beta):
    p = ExponentialThreshold(eta=0.5, beta=beta)
    x = np.random.default_rng(2).uniform(0.02, 0.98, 500)
    h = 1e-7
    fd = (p.forward(x + h) - p.forward(x - h)) / (2 * h)
    analytic = p.gradient(x)
    assert np.max(np.abs(fd - analytic)) / np.max(np.abs(analytic)) < 1e-6


def test_beta_continuation_doubles_and_saturates():
    p = ExponentialThreshold(eta=0.5, beta=99.0)
    schedule = BetaContinuation(projection=p, beta0=1.0, beta_max=8.0, interval=3)
    assert p.beta == 1.0  # reset on construction

    assert schedule.step(converged=False) is False
    assert schedule.step(converged=False) is False
    assert schedule.step(converged=False) is True and p.beta == 2.0
    # Convergence advances it early, before the interval elapses.
    assert schedule.step(converged=True) is True and p.beta == 4.0
    assert schedule.step(converged=True) is True and p.beta == 8.0
    assert schedule.at_max
    assert schedule.step(converged=True) is False and p.beta == 8.0
