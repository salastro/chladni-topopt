"""Selection vectors, mesh bookkeeping and mode tracking."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.bc import CornerClamps, DofSet
from chladni_topopt.eigen import EigenSolution
from chladni_topopt.selection import (
    build_selection_vectors,
    signed_distance_to_loops,
    widths_in_elements,
)
from chladni_topopt.tracking import MACTracker, mac
from examples.shapes import circle, letter_d, letter_t, letter_u


def test_zones_are_disjoint_and_separated_by_the_gap(mesh, dofs):
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(mesh, dofs, circle(1.0, 1.2, 0.5), band, gap, surround)

    assert not np.any(sel.shape * sel.surround)
    dist = np.abs(sel.signed_distance)
    nd = mesh.node_dofs(0)
    assert dist[sel.shape[nd] > 0].max() <= band + 1e-12
    assert dist[sel.surround[nd] > 0].min() > band + gap - 1e-12


def test_surrounding_area_covers_both_sides(mesh, dofs):
    """``L~`` must straddle the curve, or nothing forces a sign change across it."""
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(mesh, dofs, circle(1.0, 1.2, 0.5), band, gap, surround)
    nd = mesh.node_dofs(0)
    marked = sel.surround[nd] > 0
    assert np.any(marked & (sel.signed_distance < 0)), "no interior coverage"
    assert np.any(marked & (sel.signed_distance > 0)), "no exterior coverage"


def test_constrained_nodes_are_excluded(mesh, dofs):
    """Clamped nodes have zero displacement by construction, so they must not count."""
    # A circle large enough to reach into the corner clamps.
    band, gap, surround = widths_in_elements(mesh, 2, 1, 12)
    sel = build_selection_vectors(mesh, dofs, circle(1.0, 1.2, 0.9), band, gap, surround)
    assert not np.any(sel.shape[dofs.fixed])
    assert not np.any(sel.surround[dofs.fixed])


def test_only_transverse_dofs_are_selected(mesh, dofs):
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(mesh, dofs, circle(1.0, 1.2, 0.5), band, gap, surround)
    for vec in (sel.shape, sel.surround):
        marked = np.flatnonzero(vec)
        assert np.all(marked % mesh.ndof_per_node == 0)


def test_signed_distance_sign_convention():
    """Negative inside the loop, positive outside, zero on it."""
    loops = circle(0.0, 0.0, 1.0, n=720)
    points = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0], [1.0, 0.0]])
    d = signed_distance_to_loops(points, loops)
    assert d[0] == pytest.approx(-1.0, abs=1e-4)
    assert d[1] == pytest.approx(-0.5, abs=1e-4)
    assert d[2] == pytest.approx(1.0, abs=1e-4)
    assert abs(d[3]) < 1e-4


def test_empty_selection_is_rejected(mesh, dofs):
    """A band too narrow to catch any node is a configuration error, not silence."""
    with pytest.raises(ValueError, match="L~|surround"):
        build_selection_vectors(
            mesh, dofs, circle(1.0, 1.2, 0.5), band=mesh.dx, gap=10.0, surround=1e-6
        )


@pytest.mark.parametrize("shape", [letter_d, letter_t, letter_u])
def test_letters_produce_usable_zones(mesh, dofs, shape):
    band, gap, surround = widths_in_elements(mesh, 1, 1, 4)
    sel = build_selection_vectors(mesh, dofs, shape(1.0, 1.2, 1.0, 1.2), band, gap, surround)
    assert sel.n_shape > 0 and sel.n_surround > sel.n_shape


def test_mesh_connectivity_is_counter_clockwise(mesh):
    coords = mesh.node_coords()
    conn = mesh.connectivity()
    quad = coords[conn[0]]
    # Shoelace area of a counter-clockwise quad is positive.
    area = 0.5 * np.sum(
        quad[:, 0] * np.roll(quad[:, 1], -1) - np.roll(quad[:, 0], -1) * quad[:, 1]
    )
    assert area == pytest.approx(mesh.elem_area, rel=1e-12)


def test_corner_clamps_fix_four_square_patches(mesh):
    dofs = DofSet.from_mesh(mesh, CornerClamps(0.1))
    coords = mesh.node_coords()
    nodes = np.unique(dofs.fixed // mesh.ndof_per_node)
    side = 0.1 * min(mesh.lx, mesh.ly)
    per_corner = (round(side / mesh.dx) + 1) * (round(side / mesh.dy) + 1)
    assert len(nodes) == 4 * per_corner
    assert len(dofs.fixed) == len(nodes) * mesh.ndof_per_node
    for x, y in coords[nodes]:
        assert (x <= side + 1e-9 or x >= mesh.lx - side - 1e-9)
        assert (y <= side + 1e-9 or y >= mesh.ly - side - 1e-9)


def test_mac_is_scale_and_sign_invariant():
    rng = np.random.default_rng(5)
    v = rng.normal(size=50)
    candidates = np.column_stack([v, -3.0 * v, rng.normal(size=50)])
    scores = mac(v, candidates)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] < 0.5


def test_mac_tracker_follows_a_swapped_mode():
    """When two modes exchange places, the tracker must follow the shape."""
    rng = np.random.default_rng(6)
    basis = np.linalg.qr(rng.normal(size=(30, 4)))[0]

    first = EigenSolution(
        values=np.arange(4.0), reduced=basis, full=basis
    )
    tracker = MACTracker(index=2)
    assert tracker.select(first).index == 2

    swapped = basis[:, [0, 2, 1, 3]]
    second = EigenSolution(values=np.arange(4.0), reduced=swapped, full=swapped)
    result = tracker.select(second)
    assert result.index == 1
    assert result.switched
    assert result.mac == pytest.approx(1.0)


def test_linear_profile_shape_ratio_grows_with_band_width(mesh, dofs, assembler):
    """The estimate scales with the band width, quadratically.

    A wider band traps more of the mode's amplitude next to the nodal line, so
    the ratio a given design can reach rises -- fast, because the indicator is
    quadratic in the distance.
    """
    from chladni_topopt.eigen import ShiftInvertEigen
    from chladni_topopt.material import SIMPTwoMaterial
    from chladni_topopt.selection import linear_profile_shape_ratio

    material = SIMPTwoMaterial(e1=68.5e9, e2=1.0e9, rho1=2700.0, rho2=1180.0)
    x = np.full(mesh.n_elem, 0.5)
    solution = ShiftInvertEigen(n_modes=6).solve(
        assembler.stiffness(material.youngs(x)),
        assembler.mass(material.density(x)),
        dofs,
    )
    phi = solution.full[:, 4]

    floors = []
    for band in (1.0, 2.0, 3.0):
        b, g, s = widths_in_elements(mesh, band, 2, 8)
        sel = build_selection_vectors(mesh, dofs, letter_d(1.0, 1.2, 1.0, 1.2), b, g, s)
        floors.append(linear_profile_shape_ratio(mesh, sel, phi))

    assert floors == sorted(floors), f"estimate must increase with band width: {floors}"
    assert all(0.0 < f < 1.0 for f in floors)


def test_default_band_keeps_the_paper_epsilon_in_reach():
    """The shipped defaults must not pose a needlessly hard constraint.

    A 2-element band puts the estimate around 0.04, four times the paper's
    ``eps = 0.01``; the 1-element default puts it under 0.01.  The estimate is
    not a hard bound -- a design can beat it -- but a default that sits well
    above ``eps`` makes the run much harder for no reason, so this guards
    against either default drifting.
    """
    from chladni_topopt.config import (
        MeshConfig,
        ProblemConfig,
        TargetConfig,
        build_problem,
    )
    from chladni_topopt.formulations import ConstraintFormulation
    from chladni_topopt.selection import linear_profile_shape_ratio

    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=100, ny=120),
        n_modes=6,
    )
    problem = build_problem(config, verbose=False)
    state = problem.analysis.solve(problem.x0)
    estimate = linear_profile_shape_ratio(
        problem.mesh, problem.analysis.selections[0], state.modes[0].phi_full
    )
    assert estimate < ConstraintFormulation().eps, (
        f"default band gives estimate {estimate:.4f} >= eps "
        f"{ConstraintFormulation().eps}; the constraint would be needlessly hard"
    )
