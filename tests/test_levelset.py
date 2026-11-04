"""The level-set parameterisation: each piece on its own, then the whole chain.

The finite-difference check is the important one, as for the density method:
it confirms that the unchanged adjoint, fed through ``NodalToElement`` and
``WangHeaviside``, gives the derivative with respect to nodal ``Phi``.
"""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.analysis import ChladniAnalysis
from chladni_topopt.config import MeshConfig, OptimizerConfig, ProblemConfig, TargetConfig
from chladni_topopt.eigen import ShiftInvertEigen
from chladni_topopt.formulations import ConstraintFormulation
from chladni_topopt.io import load_design, save_design
from chladni_topopt.levelset import (
    ContourReinitializer,
    LevelSetConfig,
    NodalToElement,
    WangHeaviside,
    build_levelset_problem,
    frozen_nodes,
    perforated,
)
from chladni_topopt.material import SolidVoidErsatz
from chladni_topopt.metrics import nodal_line_fit
from chladni_topopt.formulations import MultiObjectiveFormulation
from chladni_topopt.levelset.initial import live_edge, target_corridor
from chladni_topopt.optimizers.hamilton_jacobi import upwind_step
from chladni_topopt.regularizers import PerimeterTV, RegularizedFormulation, count_holes
from chladni_topopt.selection import signed_distance_to_target
from chladni_topopt.tracking import FixedIndexTracker
from examples.shapes import letter_d


def _grid(mesh, values):
    return values.reshape(mesh.nx + 1, mesh.ny + 1)


def _hole_radius(mesh, phi):
    """Equivalent radius of the region ``Phi < 0``, from the element map."""
    elem = NodalToElement(mesh).forward(phi)
    return np.sqrt((elem < 0.0).sum() * mesh.elem_area / np.pi)


# -- Heaviside -----------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 1e-3])
def test_heaviside_gradient_matches_finite_differences(alpha):
    h = WangHeaviside(width=0.04, alpha=alpha)
    p = np.linspace(-0.039, 0.039, 41)
    step = 1e-7
    fd = (h.forward(p + step) - h.forward(p - step)) / (2 * step)
    assert np.allclose(h.gradient(p), fd, rtol=1e-5, atol=1e-6)


def test_heaviside_limits_and_delta_mass():
    h = WangHeaviside(width=0.04, alpha=1e-3)
    assert h.forward(np.array([-1.0]))[0] == pytest.approx(1e-3)
    assert h.forward(np.array([1.0]))[0] == pytest.approx(1.0)
    p = np.linspace(-0.05, 0.05, 20001)
    assert np.trapezoid(h.gradient(p), p) == pytest.approx(1.0 - 1e-3, rel=1e-6)
    assert np.all(h.gradient(np.array([-0.05, 0.05])) == 0.0)


# -- nodal-to-element map ----------------------------------------------------


def test_nodal_map_backward_is_the_transpose(mesh):
    rng = np.random.default_rng(0)
    m = NodalToElement(mesh)
    phi, g = rng.normal(size=mesh.n_node), rng.normal(size=mesh.n_elem)
    assert m.forward(phi) @ g == pytest.approx(phi @ m.backward(g), rel=1e-12)
    assert np.allclose(m.forward(np.ones(mesh.n_node)), 1.0)


# -- reinitialisation and the upwind step -----------------------------------


def test_reinit_restores_unit_gradient_and_keeps_the_boundary():
    from chladni_topopt.mesh import StructuredQuadMesh

    mesh = StructuredQuadMesh(2.0, 2.0, 80, 80)
    xy = mesh.node_coords()
    r = np.linalg.norm(xy - 1.0, axis=1)
    phi = 3.0 * (r - 0.5) + 0.5 * (r - 0.5) ** 2  # right zero set, wrong slope
    out = ContourReinitializer(mesh)(phi)

    assert _hole_radius(mesh, out) == pytest.approx(_hole_radius(mesh, phi), abs=0.5 * mesh.dx)
    g = np.gradient(_grid(mesh, out), mesh.dx, mesh.dy)
    slope = np.hypot(*g).ravel()
    near = (np.abs(r - 0.5) < 0.2) & (np.abs(r - 0.5) > 2 * mesh.dx)
    assert np.allclose(slope[near], 1.0, atol=0.05)


def test_upwind_step_moves_a_boundary_at_the_given_speed():
    """``dPhi/dt = |grad Phi|`` (speed 1) shrinks a hole by ``t``."""
    from chladni_topopt.mesh import StructuredQuadMesh

    mesh = StructuredQuadMesh(2.0, 2.0, 100, 100)
    r = np.linalg.norm(mesh.node_coords() - 1.0, axis=1)
    phi = _grid(mesh, r - 0.6)
    dt, steps = 0.5 * mesh.dx, 20
    for _ in range(steps):
        phi = upwind_step(phi, np.ones_like(phi), dt, mesh.dx, mesh.dy)
    radius = _hole_radius(mesh, phi.ravel())
    assert radius == pytest.approx(0.6 - steps * dt, abs=mesh.dx)
    assert np.all(np.isfinite(phi))


# -- material and initial design ----------------------------------------------


def test_perforation_and_frozen_clamps(mesh, dofs):
    phi = perforated(mesh, holes=(3, 4), radius=0.35)
    elem = NodalToElement(mesh).forward(phi)
    void = (elem < 0.0).mean()
    cell = min(2.0 / 3, 2.4 / 4)
    expected = 12 * np.pi * (0.35 * cell) ** 2 / (2.0 * 2.4)
    assert void == pytest.approx(expected, abs=0.05)

    frozen = frozen_nodes(mesh, dofs, margin=0.1)
    clamped = np.unique(dofs.fixed // mesh.ndof_per_node)
    assert set(clamped) <= set(frozen) and len(frozen) > len(clamped)


def _levelset_analysis(
    mesh, element, dofs, assembler, selection, width_elements=2.0, formulation=None
):
    return ChladniAnalysis(
        assembler=assembler,
        dofs=dofs,
        material=SolidVoidErsatz(e=68.5e9, rho=2700.0),
        filter=NodalToElement(mesh),
        projection=WangHeaviside(width=width_elements * mesh.dx),
        eigen_solver=ShiftInvertEigen(n_modes=8),
        formulation=formulation or ConstraintFormulation(eps=0.01, tau=0.4, vstar=0.55),
        selections=[selection],
        trackers=[FixedIndexTracker(index=4)],
    )


@pytest.fixture
def phi_design(mesh):
    """Perforated, with noise so no symmetry leaves repeated eigenvalues."""
    rng = np.random.default_rng(3)
    return perforated(mesh, holes=(3, 4), radius=0.35) + 0.3 * mesh.dx * rng.normal(
        size=mesh.n_node
    )


def test_ersatz_voids_carry_no_local_modes(mesh, element, dofs, assembler, selection, phi_design):
    """Holes move with the plate and hold no kinetic energy of their own.

    The test mesh is coarse: at the default two-element Heaviside width its
    holes never become fully void, so a sharper one is used.
    """
    analysis = _levelset_analysis(
        mesh, element, dofs, assembler, selection, width_elements=0.5
    )
    state = analysis.solve(phi_design)
    xbar = state.design.xbar
    assert (xbar < 1e-3).mean() > 0.15
    density = analysis.material.density(xbar)
    for k in range(state.solution.full.shape[1]):
        phi = state.solution.full[:, k]
        energy = assembler.element_quadratic(phi, phi, "m") * density
        assert energy[xbar < 1e-3].sum() / energy.sum() < 1e-3


@pytest.mark.parametrize("perimeter_weight", [0.0, 0.5])
def test_levelset_gradients_match_finite_differences(
    mesh, element, dofs, assembler, selection, phi_design, perimeter_weight
):
    """``dC/dPhi`` at nodes in the Heaviside band, through the unchanged adjoint,
    with and without the perimeter penalty on the objective."""
    formulation = ConstraintFormulation(eps=0.01, tau=0.4, vstar=0.55)
    if perimeter_weight:
        formulation = RegularizedFormulation(formulation, PerimeterTV(mesh), perimeter_weight)
    analysis = _levelset_analysis(
        mesh, element, dofs, assembler, selection, formulation=formulation
    )
    _, responses, gradients = analysis.evaluate(phi_design)

    clamped = set(np.unique(dofs.fixed // mesh.ndof_per_node))
    band = [
        n for n in np.flatnonzero(np.abs(phi_design) < 1.2 * mesh.dx) if n not in clamped
    ]
    probes = np.random.default_rng(5).choice(band, 10, replace=False)
    step = 1e-7
    for n in probes:
        pp, pm = phi_design.copy(), phi_design.copy()
        pp[n] += step
        pm[n] -= step
        fd = (
            analysis.responses(analysis.solve(pp)).values
            - analysis.responses(analysis.solve(pm)).values
        ) / (2 * step)
        scale = np.abs(gradients).max(axis=1)
        assert np.all(np.abs(fd - gradients[:, n]) <= 1e-4 * scale), (n, fd, gradients[:, n])


# -- perimeter penalty and hole count ------------------------------------------


def test_perimeter_of_a_disc_is_its_circumference():
    from chladni_topopt.mesh import StructuredQuadMesh

    mesh = StructuredQuadMesh(2.0, 2.0, 200, 200)
    r = np.linalg.norm(mesh.elem_centroids() - 1.0, axis=1)
    perimeter = PerimeterTV(mesh)
    value, _ = perimeter(np.where(r < 0.5, 0.0, 1.0))
    # Total variation on a grid measures length in the 1-norm, so a circle
    # comes out as 4/pi times its circumference -- the 8r of its bounding square.
    assert value == pytest.approx(8 * 0.5, rel=0.02)
    assert perimeter(np.ones(mesh.n_elem))[0] == pytest.approx(0.0, abs=1e-12)
    # A smoothed jump costs the same as a sharp one.
    smooth = np.clip((r - 0.45) / 0.1, 0.0, 1.0)
    assert perimeter(smooth)[0] == pytest.approx(value, rel=0.03)


def test_perimeter_gradient_matches_finite_differences(mesh):
    rng = np.random.default_rng(2)
    xbar = rng.uniform(0.0, 1.0, mesh.n_elem)
    perimeter = PerimeterTV(mesh)
    _, grad = perimeter(xbar)
    step = 1e-7
    for e in rng.choice(mesh.n_elem, 10, replace=False):
        xp, xm = xbar.copy(), xbar.copy()
        xp[e] += step
        xm[e] -= step
        fd = (perimeter(xp)[0] - perimeter(xm)[0]) / (2 * step)
        # Against the size of the whole gradient: some entries are tiny, and
        # the difference of a sum over every edge loses their low digits.
        assert abs(fd - grad[e]) <= 1e-6 * np.abs(grad).max()


@pytest.mark.parametrize(
    "inner", [ConstraintFormulation(), MultiObjectiveFormulation()], ids=["eq15", "eq21"]
)
def test_regularized_formulation_adds_only_the_penalty(
    mesh, element, dofs, assembler, selection, phi_design, inner
):
    analysis = _levelset_analysis(mesh, element, dofs, assembler, selection, formulation=inner)
    state = analysis.solve(phi_design)
    base = inner.responses(state.modes, state.design)
    wrapped = RegularizedFormulation(inner, PerimeterTV(mesh), weight=0.3)
    first = wrapped.responses(state.modes, state.design)
    p0, grad = wrapped.perimeter(state.design.xbar)

    assert wrapped.constraint_names == inner.constraint_names
    assert first.values[0] == pytest.approx(base.values[0] + 0.3)  # P / P0 = 1 at the start
    assert np.allclose(first.values[1:], base.values[1:])
    assert np.allclose(first.dxbar[0], base.dxbar[0] + 0.3 / p0 * grad)
    assert np.allclose(first.dxbar[1:], base.dxbar[1:])
    assert first.diagnostics["perimeter_ratio"] == pytest.approx(1.0)
    assert first.diagnostics["holes"] == count_holes(mesh, state.design.xbar)


def test_holes_are_counted_including_edge_notches():
    from chladni_topopt.mesh import StructuredQuadMesh

    mesh = StructuredQuadMesh(1.0, 1.0, 40, 40)
    c = mesh.elem_centroids()
    xbar = np.ones(mesh.n_elem)
    for centre in ([0.25, 0.25], [0.75, 0.25], [0.5, 0.7]):
        xbar[np.linalg.norm(c - centre, axis=1) < 0.1] = 0.0
    xbar[(c[:, 0] < 0.08) & (np.abs(c[:, 1] - 0.6) < 0.05)] = 0.0  # notch in the left edge
    assert count_holes(mesh, xbar) == 4


def test_target_corridor_stays_solid():
    """No element within band + gap of the D may turn into a hole."""
    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=50, ny=60),
        optimizer=OptimizerConfig(max_iterations=5),
    ).coarsen(1)
    problem = build_levelset_problem(
        config, LevelSetConfig(reinit_every=2), verbose=False
    )
    problem.driver.verbose = False
    mesh, target = problem.mesh, config.targets[0]
    unit = 0.5 * (mesh.dx + mesh.dy)
    reach = (target.band_elements + target.gap_elements) * unit

    corridor = target_corridor(mesh, problem.analysis.selections[0], reach)
    assert set(corridor) <= set(problem.driver.optimizer.frozen)

    result = problem.run()
    d = signed_distance_to_target(mesh.elem_centroids(), loops=list(target.loops))
    assert np.all(result.state.design.xbar[np.abs(d) <= reach] == 1.0)


def test_live_edge_puts_a_boundary_round_the_plate(mesh):
    """Edge nodes become void, the next ring in stays as it was: a zero contour
    half an element inside the edge, all the way round."""
    holes = perforated(mesh, holes=(3, 4))
    phi = live_edge(mesh, holes, margin=0.5 * mesh.dx)
    i, j = np.divmod(np.arange(mesh.n_node), mesh.ny + 1)
    ring = np.minimum.reduce([i, mesh.nx - i, j, mesh.ny - j])
    assert np.allclose(phi[ring == 0], np.minimum(holes[ring == 0], -0.5 * mesh.dx))
    # Further in, the edge adds no void: material stays material.
    assert np.all((phi > 0.0)[(ring >= 1) & (holes > 0.0)])


def test_live_edge_outline_moves_and_clamps_stay_solid():
    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=20, ny=24),
        optimizer=OptimizerConfig(max_iterations=6),
    )
    problem = build_levelset_problem(
        config,
        LevelSetConfig(holes=(3, 4), live_edge=True, perimeter_weight=0.0, protect_target=False),
        verbose=False,
    )
    problem.driver.verbose = False
    phi0 = problem.x0
    edge = np.flatnonzero(np.isclose(problem.mesh.node_coords()[:, 0], 0.0))
    result = problem.run()
    assert not np.allclose(result.x[edge], phi0[edge]), "the outline did not move"
    clamps = frozen_nodes(problem.mesh, problem.dofs)
    assert np.all(result.x[clamps] > 0.0)


def _domain_problem(iterations=4, nx_scale=1):
    from chladni_topopt.domain import PlateInDomain

    domain = PlateInDomain(plate=(2.0, 2.4), margin=0.3, element=0.1 / nx_scale)
    config = ProblemConfig(
        targets=[TargetConfig(loops=tuple(domain.shift(letter_d(1.0, 1.2, 1.0, 1.2))), mode_index=4)],
        mesh=domain.mesh(),
        boundary=domain.clamps(),
        formulation=ConstraintFormulation(vstar=None),
        optimizer=OptimizerConfig(max_iterations=iterations),
    )
    levelset = LevelSetConfig(domain=domain, holes=(3, 4), width=0.2, step=0.05, step_min=0.01)
    return build_levelset_problem(config, levelset, verbose=False), domain


def test_domain_start_has_the_plate_outline_as_a_zero_contour():
    problem, domain = _domain_problem()
    phi = problem.x0
    plate = domain.plate_sdf(problem.mesh.node_coords())
    # Frozen nodes (the clamps' surroundings, the target corridor) are held solid.
    free = ~np.isin(np.arange(len(phi)), problem.driver.optimizer.frozen)
    assert np.all(phi[free & (plate < -0.05)] < 0.0)  # the margin is void
    assert np.all(phi[free & (np.abs(plate) < 1e-9)] <= 1e-12)  # outline: Phi = 0


def test_lengths_are_metres_not_elements():
    """The same LevelSetConfig gives the same physical Heaviside width on two meshes."""
    coarse, _ = _domain_problem(nx_scale=1)
    fine, _ = _domain_problem(nx_scale=2)
    assert coarse.analysis.projection.width == fine.analysis.projection.width == 0.2
    cfl_coarse = coarse.driver.optimizer.inner.cfl * coarse.mesh.dx
    cfl_fine = fine.driver.optimizer.inner.cfl * fine.mesh.dx
    assert cfl_coarse == pytest.approx(cfl_fine)


def test_outline_can_grow_into_the_margin():
    """A constant outward speed moves the plate edge into the void margin."""
    problem, domain = _domain_problem()
    hj = problem.driver.optimizer.inner
    phi = problem.x0
    grow = -np.ones_like(phi)  # G = -1 everywhere: speed +1, material advances
    moved = hj.update(phi, 0.0, grow, np.zeros(2), np.zeros((2, len(phi))))
    plate = domain.plate_sdf(problem.mesh.node_coords())
    free = ~np.isin(np.arange(len(phi)), problem.driver.optimizer.frozen)
    margin = free & (plate < 0.0) & (plate > -0.1)
    # Upwinding leaves local maxima alone, e.g. a one-node ridge between a
    # starting hole and the margin, so "nearly all" rather than all.
    assert np.mean(moved[margin] > phi[margin]) > 0.95


# -- metric ---------------------------------------------------------------------


def test_nodal_lines_in_holes_do_not_count(mesh, dofs):
    phi = np.zeros(mesh.n_dof)
    xy = mesh.node_coords()
    phi[mesh.node_dofs(0)] = xy[:, 1] - 1.2
    target = [np.array([[0.0, 1.2], [mesh.lx, 1.2]])]
    solid = np.ones(mesh.n_elem)
    solid[mesh.elem_centroids()[:, 0] < 1.0] = 0.0  # left half is a hole

    full = nodal_line_fit(mesh, phi, open_curves=target, dofs=dofs)
    masked = nodal_line_fit(mesh, phi, open_curves=target, dofs=dofs, solid=solid)
    assert full.traced == pytest.approx(1.0)
    assert masked.traced == pytest.approx(0.5, abs=0.06)


# -- end to end ---------------------------------------------------------------------


@pytest.mark.parametrize("update", ["hj", "adamw"])
def test_levelset_problem_runs_and_saves(tmp_path, update):
    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=20, ny=24),
        optimizer=OptimizerConfig(max_iterations=6),
    )
    problem = build_levelset_problem(
        config, LevelSetConfig(update=update, holes=(3, 4), reinit_every=3), verbose=False
    )
    problem.driver.verbose = False
    phi0 = problem.x0
    assert phi0.shape == (problem.mesh.n_node,)
    result = problem.run()

    assert result.iterations == 6
    assert not np.allclose(result.x, phi0), "the boundary did not move"
    frozen = frozen_nodes(problem.mesh, problem.dofs)
    assert np.all(result.x[frozen] > 0.0), "a clamp ended up in a hole"
    assert np.isnan(result.history.records[-1].beta)
    assert "holes" in result.history.records[-1].diagnostics

    data = load_design(save_design(tmp_path / "d.npz", result, problem.mesh, method="levelset"))
    assert str(data["method"]) == "levelset"
    assert data["x"].shape == phi0.shape
