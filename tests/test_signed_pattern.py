"""The scale-free, sign-aware loss: invariances, its ideal case, and gradients."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from chladni_topopt.formulations import SignedPatternFormulation
from tests.conftest import make_analysis
from tests.test_levelset import _levelset_analysis, phi_design  # noqa: F401  (fixture)
from tests.test_sensitivity import _fd_vs_adjoint, _probe_elements


def _state(analysis, design):
    return analysis.solve(design)


def test_unchanged_by_scaling_or_flipping_the_mode(mesh, element, dofs, assembler, selection, design):
    formulation = SignedPatternFormulation()
    analysis = make_analysis(mesh, element, dofs, assembler, selection, formulation=formulation)
    state = _state(analysis, design)
    base = formulation.responses(state.modes, state.design).values
    for factor in (3.0, -1.0, -0.25):
        modes = [replace(m, phi_full=factor * m.phi_full) for m in state.modes]
        assert np.allclose(formulation.responses(modes, state.design).values, base, rtol=1e-9)


def test_a_line_exactly_on_the_target_scores_full_sign_agreement(
    mesh, element, dofs, assembler, selection, design
):
    """A field equal to the signed distance has its zero line on the target."""
    formulation = SignedPatternFormulation()
    analysis = make_analysis(mesh, element, dofs, assembler, selection, formulation=formulation)
    state = _state(analysis, design)
    phi = np.zeros(mesh.n_dof)
    phi[mesh.node_dofs(0)] = selection.signed_distance
    modes = [replace(state.modes[0], phi_full=phi)]
    responses = formulation.responses(modes, state.design)
    assert responses.diagnostics["sign_agreement"] == pytest.approx(1.0)
    assert responses.values[1] == pytest.approx(-formulation.kappa)


def test_a_mode_escaping_the_zone_breaks_the_guard(mesh, element, dofs, assembler, selection, design):
    formulation = SignedPatternFormulation(eta_min=0.5)
    analysis = make_analysis(mesh, element, dofs, assembler, selection, formulation=formulation)
    state = _state(analysis, design)
    formulation.responses(state.modes, state.design)  # records eta0
    zone = (selection.shape + selection.surround) > 0
    phi = state.modes[0].phi_full.copy()
    phi[mesh.node_dofs(0)[~zone[mesh.node_dofs(0)]]] *= 50.0  # all motion outside the zone
    modes = [replace(state.modes[0], phi_full=phi)]
    assert formulation.responses(modes, state.design).values[2] > 0.0


@pytest.mark.parametrize("vstar", [None, 0.55])
def test_gradients_match_finite_differences(
    mesh, element, dofs, assembler, selection, design, vstar
):
    analysis = make_analysis(
        mesh, element, dofs, assembler, selection,
        formulation=SignedPatternFormulation(vstar=vstar),
    )
    probes = _probe_elements(mesh.n_elem, dofs, assembler)
    fd, adjoint, names = _fd_vs_adjoint(analysis, design, probes)
    for row, name in enumerate(names):
        scale = max(np.abs(adjoint[row]).max(), np.abs(fd[row]).max())
        assert np.abs(fd[row] - adjoint[row]).max() / scale < 1e-4, name


def test_level_set_gradients_match_finite_differences(
    mesh, element, dofs, assembler, selection, phi_design
):
    analysis = _levelset_analysis(
        mesh, element, dofs, assembler, selection, formulation=SignedPatternFormulation()
    )
    _, responses, gradients = analysis.evaluate(phi_design)
    clamped = set(np.unique(dofs.fixed // mesh.ndof_per_node))
    band = [n for n in np.flatnonzero(np.abs(phi_design) < 1.2 * mesh.dx) if n not in clamped]
    step = 1e-7
    for n in np.random.default_rng(5).choice(band, 8, replace=False):
        pp, pm = phi_design.copy(), phi_design.copy()
        pp[n] += step
        pm[n] -= step
        fd = (
            analysis.responses(analysis.solve(pp)).values
            - analysis.responses(analysis.solve(pm)).values
        ) / (2 * step)
        scale = np.abs(gradients).max(axis=1)
        assert np.all(np.abs(fd - gradients[:, n]) <= 1e-4 * scale), (n, fd, gradients[:, n])
