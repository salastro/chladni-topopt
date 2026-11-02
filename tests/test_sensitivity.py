"""Finite-difference verification of the bordered-adjoint sensitivities.

This is the test that pins down Eqs. (16)-(18), including the factor
conventions on ``u`` and ``v`` that the paper and this implementation scale
differently.  Everything else in the pipeline can be checked by inspection;
the adjoint cannot.

Errors are measured against the size of the whole gradient vector, not
elementwise.  Elements inside the clamped corners have an exactly zero
gradient -- every DOF they touch is constrained -- and an elementwise relative
error would report a meaningless 100% there while both quantities are zero.

The gradient tests run once per available factorization backend.  With
PARDISO the bordered adjoint goes through Bunch-Kaufman pivoting with pivot
perturbation; this is the check that would catch that degrading the gradients,
or a matrix declared with the wrong type.
"""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.formulations import ConstraintFormulation, MultiObjectiveFormulation
from chladni_topopt.indicators import AVERAGE, VARIANCE
from tests.conftest import make_analysis

PROBE_COUNT = 12
STEP = 1e-6


def _probe_elements(n_elem: int, dofs, assembler) -> np.ndarray:
    """Elements with at least one free DOF, so the gradient can be nonzero."""
    fixed = np.zeros(dofs.n_dof, dtype=bool)
    fixed[dofs.fixed] = True
    free_per_elem = (~fixed[assembler.element_dofs]).sum(axis=1)
    candidates = np.flatnonzero(free_per_elem == assembler.element_dofs.shape[1])
    return np.random.default_rng(7).choice(candidates, PROBE_COUNT, replace=False)


def _fd_vs_adjoint(analysis, x, probes):
    _, responses, gradients = analysis.evaluate(x)
    fd = np.zeros((responses.n_responses, len(probes)))
    for j, e in enumerate(probes):
        xp, xm = x.copy(), x.copy()
        xp[e] += STEP
        xm[e] -= STEP
        fp = analysis.responses(analysis.solve(xp)).values
        fm = analysis.responses(analysis.solve(xm)).values
        fd[:, j] = (fp - fm) / (2 * STEP)
    return fd, gradients[:, probes], responses.names


@pytest.mark.parametrize("beta", [1.0, 16.0])
def test_constraint_formulation_gradients(
    mesh, element, dofs, assembler, selection, design, beta, factorizer
):
    analysis = make_analysis(
        mesh, element, dofs, assembler, selection,
        formulation=ConstraintFormulation(eps=0.01, tau=0.4, vstar=0.55), beta=beta,
        factorizer=factorizer,
    )
    probes = _probe_elements(mesh.n_elem, dofs, assembler)
    fd, adjoint, names = _fd_vs_adjoint(analysis, design, probes)

    for row, name in enumerate(names):
        scale = max(np.abs(adjoint[row]).max(), np.abs(fd[row]).max())
        error = np.abs(fd[row] - adjoint[row]).max() / scale
        assert error < 1e-4, f"{name}: relative error {error:.2e}"


def test_multi_objective_formulation_gradients(
    mesh, element, dofs, assembler, selection, design, factorizer
):
    analysis = make_analysis(
        mesh, element, dofs, assembler, selection,
        formulation=MultiObjectiveFormulation(), beta=4.0,
        factorizer=factorizer,
    )
    probes = _probe_elements(mesh.n_elem, dofs, assembler)
    fd, adjoint, names = _fd_vs_adjoint(analysis, design, probes)

    for row, name in enumerate(names):
        scale = max(np.abs(adjoint[row]).max(), np.abs(fd[row]).max())
        error = np.abs(fd[row] - adjoint[row]).max() / scale
        assert error < 1e-4, f"{name}: relative error {error:.2e}"


def test_volume_gradient_is_exact(mesh, element, dofs, assembler, selection, design):
    """The volume constraint has no eigenvector dependence; its adjoint is zero."""
    analysis = make_analysis(mesh, element, dofs, assembler, selection)
    state, responses, gradients = analysis.evaluate(design)
    dxbar = analysis.sensitivity.dc_dxbar(
        responses, state.modes, state.design, state.stiffness, state.mass
    )
    # Row 3 is the volume constraint; its dc/dxbar must equal the direct part.
    assert np.allclose(dxbar[3], responses.dxbar[3])
    assert np.all(gradients[3] > 0.0)


def test_indicator_gradients_match_finite_differences(selection):
    """``d Avg/d phi`` and ``d Var/d phi`` in isolation."""
    rng = np.random.default_rng(3)
    phi = rng.normal(size=len(selection.shape))
    h = 1e-6
    for indicator, sel in (
        (AVERAGE, selection.shape),
        (VARIANCE, selection.surround),
        (AVERAGE, selection.surround),
    ):
        analytic = indicator.gradient(sel, phi)
        probes = np.flatnonzero(sel)[:: max(len(np.flatnonzero(sel)) // 20, 1)]
        for i in probes:
            pp, pm = phi.copy(), phi.copy()
            pp[i] += h
            pm[i] -= h
            fd = (indicator.value(sel, pp) - indicator.value(sel, pm)) / (2 * h)
            assert fd == pytest.approx(analytic[i], rel=1e-5, abs=1e-12)


def test_variance_gradient_drops_the_mean_term(selection):
    """``sum_i L_i (phi_i^2 - mu) == 0`` is what lets ``d mu/d phi`` cancel."""
    phi = np.random.default_rng(4).normal(size=len(selection.surround))
    sel = selection.surround
    mu = AVERAGE.value(sel, phi)
    assert sel @ (phi * phi - mu) == pytest.approx(0.0, abs=1e-9 * sel.sum())
