"""The spline outline: geometry, the signed-distance map, and the whole chain."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.config import OptimizerConfig, ProblemConfig, TargetConfig
from chladni_topopt.domain import PlateInDomain
from chladni_topopt.formulations import ConstraintFormulation
from chladni_topopt.selection import signed_distance_to_target
from chladni_topopt.spline import (
    ClosedBSpline,
    SplineConfig,
    build_spline_problem,
    rectangle_controls,
)
from examples.shapes import letter_d

DOMAIN = PlateInDomain(plate=(2.0, 2.4), margin=0.3, element=0.1)


def _problem(perimeter_weight=0.2, protect_target=True, iterations=3):
    config = ProblemConfig(
        targets=[
            TargetConfig(loops=tuple(DOMAIN.shift(letter_d(1.0, 1.2, 1.0, 1.2))), mode_index=4)
        ],
        mesh=DOMAIN.mesh(),
        boundary=DOMAIN.clamps(),
        formulation=ConstraintFormulation(vstar=None),
        optimizer=OptimizerConfig(method="mma", max_iterations=iterations, move=0.05),
    )
    spline = SplineConfig(
        domain=DOMAIN, n_ctrl=32, width=0.2,
        perimeter_weight=perimeter_weight, protect_target=protect_target,
    )
    return build_spline_problem(config, spline, verbose=False)


def test_basis_is_a_partition_of_unity_and_closed():
    spline = ClosedBSpline(n_ctrl=12, samples=600)
    assert np.allclose(np.asarray(spline.basis.sum(axis=1)).ravel(), 1.0)
    assert np.allclose(np.asarray(spline.tangent.sum(axis=1)).ravel(), 0.0, atol=1e-12)
    # Periodic: the curve's end runs back into its start.
    points = np.column_stack([np.cos(np.arange(12)), np.sin(np.arange(12) * 1.3)])
    curve = spline.basis @ points
    step = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    assert np.linalg.norm(curve[0] - curve[-1]) < 3 * step.max()


def test_rectangle_controls_reproduce_the_plate():
    controls, dirs = rectangle_controls(0.3, 0.3, 2.0, 2.4, 48)
    assert len(controls) == 48
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    curve = ClosedBSpline(48).basis @ controls
    sd = DOMAIN.plate_sdf(curve)
    # On the plate edge along the straight runs; corners rounded slightly inside.
    assert np.max(np.abs(sd)) < 0.1
    assert np.median(np.abs(sd)) < 1e-9


def test_zero_offsets_start_from_the_plate():
    problem = _problem()
    spline_map = problem.analysis.filter
    assert np.allclose(spline_map.offsets(problem.x0), 0.0)
    sd = spline_map.forward(problem.x0)
    inside = DOMAIN.plate_sdf(problem.mesh.elem_centroids()) > 0.15
    assert np.all(sd[inside] > 0.0)


def test_signed_distance_jacobian_matches_finite_differences():
    problem = _problem()
    spline_map = problem.analysis.filter
    u = problem.x0 + np.random.default_rng(0).uniform(-0.05, 0.05, spline_map.n)
    sd = spline_map.forward(u)
    jac = spline_map._jacobian.toarray()
    band = np.abs(sd) < spline_map.band
    h = 1e-6
    for j in (0, 7, 19):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        fd = (spline_map.forward(up) - spline_map.forward(um)) / (2 * h)
        assert np.abs(fd - jac[:, j])[band].max() <= 1e-6 * np.abs(jac[:, j]).max() + 1e-9


@pytest.mark.parametrize("perimeter_weight", [0.0, 0.5])
def test_chain_gradient_matches_finite_differences(perimeter_weight):
    problem = _problem(perimeter_weight=perimeter_weight)
    analysis = problem.analysis
    u = problem.x0 + np.random.default_rng(1).uniform(-0.03, 0.03, analysis.filter.n)
    _, responses, gradients = analysis.evaluate(u)
    h = 1e-6
    for j in (2, 11, 25):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        fd = (
            analysis.responses(analysis.solve(up)).values
            - analysis.responses(analysis.solve(um)).values
        ) / (2 * h)
        analysis.evaluate(u)  # restore the map's Jacobian to u
        scale = np.abs(gradients).max(axis=1) + 1e-12
        assert np.all(np.abs(fd - gradients[:, j]) <= 1e-4 * scale), (j, fd, gradients[:, j])


def test_corridor_stays_solid_whatever_the_outline():
    problem = _problem()
    spline_map = problem.analysis.filter
    sd = spline_map.forward(np.zeros(spline_map.n))  # every point pulled fully inward
    loops = list(problem.config.targets[0].loops)
    d = signed_distance_to_target(problem.mesh.elem_centroids(), loops)
    target = problem.config.targets[0]
    reach = (target.band_elements + target.gap_elements) * 0.1
    assert np.all(sd[np.abs(d) <= reach] >= spline_map.frozen_value)


def test_mma_run_moves_the_outline():
    problem = _problem(iterations=3)
    problem.driver.verbose = False
    result = problem.run()
    assert result.iterations == 3
    assert not np.allclose(result.x, problem.x0)
    assert "perimeter_ratio" in result.history.records[-1].diagnostics
