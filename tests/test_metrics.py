"""The nodal-line fit metric, on fields whose nodal lines are known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.metrics import nodal_line_fit, nodal_lines


def _field(mesh, dofs, fn):
    """A full-length vector whose transverse DOFs are ``fn(x, y)``, zero where clamped."""
    phi = np.zeros(mesh.n_dof)
    xy = mesh.node_coords()
    phi[mesh.node_dofs(0)] = fn(xy[:, 0], xy[:, 1])
    phi[dofs.fixed] = 0.0
    return phi


def test_a_line_on_the_target_is_fully_traced(mesh, dofs):
    phi = _field(mesh, dofs, lambda x, y: y - 1.2)
    target = [np.array([[0.0, 1.2], [mesh.lx, 1.2]])]
    fit = nodal_line_fit(mesh, phi, open_curves=target, dofs=dofs)
    assert fit.traced == pytest.approx(1.0)
    assert fit.stray == pytest.approx(0.0)
    assert fit.mean_gap < 0.01


def test_a_line_elsewhere_traces_nothing(mesh, dofs):
    phi = _field(mesh, dofs, lambda x, y: y - 1.2)
    target = [np.array([[0.0, 0.6], [mesh.lx, 0.6]])]
    fit = nodal_line_fit(mesh, phi, open_curves=target, dofs=dofs)
    assert fit.traced == pytest.approx(0.0)
    assert fit.mean_gap == pytest.approx(0.6, abs=0.02)


def test_a_crossing_line_counts_as_stray(mesh, dofs):
    """Target y = 1.2; the field also has a nodal line x = 1.0 crossing it."""
    phi = _field(mesh, dofs, lambda x, y: (y - 1.2) * (x - 1.0))
    target = [np.array([[0.0, 1.2], [mesh.lx, 1.2]])]
    fit = nodal_line_fit(mesh, phi, open_curves=target, dofs=dofs, neighbourhood=0.25)
    # Not exactly 1: where the two nodal lines cross there is a saddle, and on
    # this coarse (0.1 m) test mesh the contour cuts the corner there, leaving
    # a few target points just outside ``tol``.
    assert fit.traced >= 0.95
    # About 2 * (0.25 - 0.06) m of the vertical line is near the target but off it.
    assert fit.stray == pytest.approx(2 * (0.25 - 0.06) / mesh.lx, rel=0.3)


def test_clamp_outlines_are_not_nodal_lines(mesh, dofs):
    """Zero displacement in the clamped corners must not register as lines."""
    phi = _field(mesh, dofs, lambda x, y: 1.0 + 0.0 * x)  # no real nodal line
    assert nodal_lines(mesh, phi) != []  # raw contour does trace the clamps
    assert nodal_lines(mesh, phi, dofs) == []


def test_sweep_mode_lists_parse():
    from examples.sweep_modes import parse_modes

    assert parse_modes("1-4") == [1, 2, 3, 4]
    assert parse_modes("10,3,6") == [3, 6, 10]
    assert parse_modes("2-4,9,3") == [2, 3, 4, 9]
