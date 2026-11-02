"""Open target curves: nodal lines that end on the plate boundary.

Before open curves existed, every target was treated as a closed loop: a
closing segment was added silently, and "which side of the line" meant
inside/outside.  These tests pin down that an open curve gets neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.config import MeshConfig, ProblemConfig, TargetConfig, build_problem
from chladni_topopt.io import load_design, save_design
from chladni_topopt.selection import (
    build_selection_vectors,
    signed_distance_to_target,
    widths_in_elements,
)
from examples.shapes import letter_d, letter_s, u_curve
from tests.conftest import make_analysis

PLATE = (2.0, 2.4)


@pytest.fixture(scope="module")
def s_curve():
    return letter_s(1.0, 1.2, 0.4, PLATE)


def test_open_curve_gets_no_closing_segment():
    """A U-shaped polyline: open, the gap at the top stays a gap."""
    u = [np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])]
    point = np.array([[0.5, 1.0]])  # right on the would-be closing segment
    as_loop = signed_distance_to_target(point, loops=u)
    as_curve = signed_distance_to_target(point, open_curves=u)
    assert abs(as_loop[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(as_curve[0]) == pytest.approx(0.5, abs=1e-12)


def test_open_curve_sign_is_the_side_of_the_line():
    """Walking left to right, points above are on the left: negative."""
    line = [np.array([[0.0, 0.5], [1.0, 0.5]])]
    points = np.array([[0.3, 0.8], [0.7, 0.1]])
    d = signed_distance_to_target(points, open_curves=line)
    assert d[0] == pytest.approx(-0.3)
    assert d[1] == pytest.approx(0.4)


def test_a_target_needs_something():
    with pytest.raises(ValueError, match="loop or open curve"):
        signed_distance_to_target(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="loop or open curve"):
        TargetConfig(loops=(), mode_index=2)


def test_letter_s_runs_edge_to_edge_smoothly(s_curve):
    curve = s_curve[0]
    assert curve[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert curve[-1, 0] == pytest.approx(PLATE[0], abs=1e-12)
    # Clear of the 0.2 m clamped corner squares.
    for end in (curve[0], curve[-1]):
        assert 0.2 < end[1] < PLATE[1] - 0.2
    # Point-symmetric about the centre, like the letter.
    assert np.allclose(curve[::-1], 2.0 * np.array([1.0, 1.2]) - curve, atol=1e-12)
    # Smooth: no cusp where an arc meets its tail.
    seg = np.diff(curve, axis=0)
    heading = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    turn = np.abs((np.diff(heading) + 180.0) % 360.0 - 180.0)
    assert turn.max() < 5.0


def test_selection_straddles_an_open_curve(mesh, dofs, s_curve):
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(
        mesh, dofs, (), band, gap, surround, open_curves=s_curve
    )
    assert sel.loops == () and len(sel.open_curves) == 1
    assert not np.any(sel.shape * sel.surround)
    nd = mesh.node_dofs(0)
    marked = sel.surround[nd] > 0
    assert np.any(marked & (sel.signed_distance < 0))
    assert np.any(marked & (sel.signed_distance > 0))


def test_open_curve_zones_reach_the_plate_edges(mesh, dofs, s_curve):
    """The band must follow the curve all the way out, not stop short."""
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(
        mesh, dofs, (), band, gap, surround, open_curves=s_curve
    )
    coords = mesh.node_coords()
    shape_nodes = coords[sel.shape[mesh.node_dofs(0)] > 0]
    assert shape_nodes[:, 0].min() == pytest.approx(0.0)
    assert shape_nodes[:, 0].max() == pytest.approx(mesh.lx)


def test_loops_and_open_curves_combine(mesh, dofs, s_curve):
    """A target may mix both; each point takes the nearer part's sign."""
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    ring = [np.column_stack([1.0 + 0.15 * np.cos(t), 2.1 + 0.1 * np.sin(t)])
            for t in [np.linspace(0, 2 * np.pi, 60, endpoint=False)]]
    both = build_selection_vectors(
        mesh, dofs, ring, band, gap, surround, open_curves=s_curve
    )
    curve_only = build_selection_vectors(
        mesh, dofs, (), band, gap, surround, open_curves=s_curve
    )
    assert both.n_shape > curve_only.n_shape


def test_open_target_builds_and_differentiates(mesh, element, dofs, assembler, s_curve):
    """Nothing downstream assumed closed loops: the adjoint still matches FD."""
    band, gap, surround = widths_in_elements(mesh, 1, 2, 4)
    sel = build_selection_vectors(
        mesh, dofs, (), band, gap, surround, open_curves=s_curve
    )
    analysis = make_analysis(mesh, element, dofs, assembler, sel, mode_index=2)
    x = np.random.default_rng(0).uniform(0.35, 0.65, mesh.n_elem)
    _, responses, gradients = analysis.evaluate(x)
    h = 1e-6
    for e in (250, 260, 270):
        xp, xm = x.copy(), x.copy()
        xp[e] += h
        xm[e] -= h
        fd = (
            analysis.responses(analysis.solve(xp)).values
            - analysis.responses(analysis.solve(xm)).values
        ) / (2 * h)
        scale = np.maximum(np.abs(gradients[:, e]), 1e-12)
        assert np.max(np.abs(fd - gradients[:, e]) / scale) < 1e-3


def test_open_curves_survive_save_and_load(tmp_path, s_curve):
    config = ProblemConfig(
        targets=[TargetConfig(loops=(), open_curves=s_curve, mode_index=2)],
        mesh=MeshConfig(nx=20, ny=24),
    )
    problem = build_problem(config, verbose=False)
    problem.driver.max_iterations = 1
    problem.driver.verbose = False
    result = problem.run()
    path = save_design(tmp_path / "design.npz", result, problem.mesh)
    data = load_design(path)
    assert data["loops"] == []
    assert len(data["curves"]) == 1
    assert np.allclose(data["curves"][0], s_curve[0])


def test_archives_without_curves_still_load(tmp_path):
    """Designs saved before open curves existed have no ``curves`` key."""
    path = tmp_path / "old.npz"
    loop = letter_d(1.0, 1.2, 1.0, 1.2)[0]
    np.savez(path, loops=loop, loop_offsets=np.array([0, len(loop)]))
    data = load_design(path)
    assert data["curves"] == []
    assert np.allclose(data["loops"][0], loop)


def test_open_curves_are_drawn_open(mesh, s_curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chladni_topopt.postprocess import plot_mode

    fig, ax = plt.subplots()
    phi = np.random.default_rng(1).normal(size=mesh.n_dof)
    plot_mode(mesh, phi, ax=ax, target_curves=s_curve)
    drawn = [line.get_xydata() for line in ax.get_lines()]
    assert len(drawn) == 1
    assert not np.allclose(drawn[0][0], drawn[0][-1])
    assert ax.get_xlim() == (0.0, mesh.lx)
    plt.close(fig)


def test_u_curve_hangs_from_the_top_edge():
    """Both arms end on the top edge, clear of the clamps; the curve is smooth."""
    curve = u_curve(1.0, 1.2, 1.0, PLATE)[0]
    for end in (curve[0], curve[-1]):
        assert end[1] == pytest.approx(PLATE[1], abs=1e-12)
        assert 0.2 < end[0] < PLATE[0] - 0.2
    assert curve[:, 1].min() == pytest.approx(1.0, abs=1e-3)
    # Mirror-symmetric about x = cx.
    mirrored = np.column_stack([2.0 - curve[::-1, 0], curve[::-1, 1]])
    assert np.allclose(mirrored, curve, atol=1e-12)
    seg = np.diff(curve, axis=0)
    heading = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
    turn = np.abs((np.diff(heading) + 180.0) % 360.0 - 180.0)
    assert turn.max() < 5.0


def test_u_selection_lines_both_arms_and_the_bottom(mesh, dofs):
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(
        mesh, dofs, (), band, gap, surround,
        open_curves=u_curve(1.0, 1.2, 1.0, PLATE),
    )
    coords = mesh.node_coords()
    shape_nodes = coords[sel.shape[mesh.node_dofs(0)] > 0]
    # The band reaches the top edge and the bottom of the U.
    assert shape_nodes[:, 1].max() == pytest.approx(mesh.ly)
    assert shape_nodes[:, 1].min() < 1.1
    # Inside and outside the U both carry surrounding nodes.
    marked = sel.surround[mesh.node_dofs(0)] > 0
    assert np.any(marked & (sel.signed_distance < 0))
    assert np.any(marked & (sel.signed_distance > 0))
