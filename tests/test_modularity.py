"""The protocols are real: components can be swapped without touching the driver.

This is the property the package is built for, so it is asserted rather than
assumed.  Each test replaces one stage of the pipeline with a different
implementation -- including one defined here, outside the package -- and checks
the rest still runs end to end.
"""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.driver import TopologyOptimizer, WindowedConvergence
from chladni_topopt.eigen import EigenSolver, ShiftInvertEigen
from chladni_topopt.elements.base import ElementFormulation
from chladni_topopt.elements.mindlin_q4 import MindlinQ4
from chladni_topopt.filters import (
    BetaContinuation,
    Continuation,
    DensityFilter,
    ExponentialThreshold,
    Filter,
    IdentityFilter,
    NoContinuation,
    Projection,
)
from chladni_topopt.formulations import (
    ConstraintFormulation,
    Formulation,
    MultiObjectiveFormulation,
)
from chladni_topopt.levelset import (
    ContourReinitializer,
    LevelSetUpdate,
    NodalToElement,
    Reinitializer,
    WangHeaviside,
)
from chladni_topopt.material import MaterialInterpolation, SIMPTwoMaterial, SolidVoidErsatz
from chladni_topopt.mesh import StructuredQuadMesh
from chladni_topopt.optimizers.hamilton_jacobi import HamiltonJacobi
from chladni_topopt.regularizers import PerimeterTV, RegularizedFormulation
from chladni_topopt.spline import ClosedBSpline, SplineMap, rectangle_controls
from chladni_topopt.optimizers.base import Optimizer
from chladni_topopt.optimizers.adamw import AdamW
from chladni_topopt.optimizers.mma import MMA
from chladni_topopt.tracking import FixedIndexTracker, MACTracker, ModeTracker
from tests.conftest import make_analysis


@pytest.mark.parametrize(
    "instance,protocol",
    [
        (MindlinQ4(thickness=0.2), ElementFormulation),
        (ShiftInvertEigen(), EigenSolver),
        (MACTracker(index=4), ModeTracker),
        (FixedIndexTracker(index=4), ModeTracker),
        (ConstraintFormulation(), Formulation),
        (MultiObjectiveFormulation(), Formulation),
        (ExponentialThreshold(), Projection),
        (IdentityFilter(), Filter),
        (SIMPTwoMaterial(1.0, 1.0, 1.0, 1.0), MaterialInterpolation),
        (MMA(n=3, m=1), Optimizer),
        (AdamW(n=3, m=1), Optimizer),
        (WangHeaviside(width=0.04), Projection),
        (NodalToElement(StructuredQuadMesh(1.0, 1.0, 2, 2)), Filter),
        (SolidVoidErsatz(1.0, 1.0), MaterialInterpolation),
        (HamiltonJacobi(grid=(3, 3), dx=0.5, dy=0.5, m=1), Optimizer),
        (LevelSetUpdate(inner=AdamW(n=9, m=1), reinit=None), Optimizer),
        (ContourReinitializer(StructuredQuadMesh(1.0, 1.0, 2, 2)), Reinitializer),
        (BetaContinuation(ExponentialThreshold()), Continuation),
        (NoContinuation(), Continuation),
        (
            SplineMap(
                ClosedBSpline(8), *rectangle_controls(0.0, 0.0, 1.0, 1.0, 8),
                t_lo=-0.1, t_hi=0.1, centroids=np.zeros((1, 2)), band=0.1,
            ),
            Filter,
        ),
        (
            RegularizedFormulation(
                ConstraintFormulation(), PerimeterTV(StructuredQuadMesh(1.0, 1.0, 2, 2))
            ),
            Formulation,
        ),
    ],
)
def test_shipped_implementations_satisfy_their_protocol(instance, protocol):
    assert isinstance(instance, protocol)


class SquaredDensityFilter:
    """A filter defined outside the package, to prove nothing is hard-wired.

    Applies the usual cone filter twice, which is a legitimate (smoother)
    alternative, and pulls sensitivities back through both passes.
    """

    def __init__(self, rmin, centroids):
        self._inner = DensityFilter(rmin=rmin, centroids=centroids)

    def forward(self, x):
        return self._inner.forward(self._inner.forward(x))

    def backward(self, dc_dxtilde):
        return self._inner.backward(self._inner.backward(dc_dxtilde))


def test_a_foreign_filter_drops_straight_in(mesh, element, dofs, assembler, selection, design):
    """A user-supplied Filter must work, including through the sensitivity chain."""
    custom = SquaredDensityFilter(rmin=2.0 * mesh.dx, centroids=mesh.elem_centroids())
    assert isinstance(custom, Filter)

    analysis = make_analysis(
        mesh, element, dofs, assembler, selection, density_filter=custom
    )
    state, responses, gradients = analysis.evaluate(design)
    assert gradients.shape == (4, mesh.n_elem)
    assert np.all(np.isfinite(gradients))

    # Double filtering really is smoother than a single pass.
    single = DensityFilter(rmin=2.0 * mesh.dx, centroids=mesh.elem_centroids())
    assert np.std(custom.forward(design)) < np.std(single.forward(design))


def test_gradients_stay_correct_after_swapping_the_filter(
    mesh, element, dofs, assembler, selection, design
):
    """A swapped component must not silently break the chain rule."""
    custom = SquaredDensityFilter(rmin=2.0 * mesh.dx, centroids=mesh.elem_centroids())
    analysis = make_analysis(
        mesh, element, dofs, assembler, selection, density_filter=custom
    )
    _, responses, gradients = analysis.evaluate(design)
    probes = np.array([300, 301, 302])
    h = 1e-6
    for e in probes:
        xp, xm = design.copy(), design.copy()
        xp[e] += h
        xm[e] -= h
        fd = (
            analysis.responses(analysis.solve(xp)).values
            - analysis.responses(analysis.solve(xm)).values
        ) / (2 * h)
        for row in range(responses.n_responses):
            scale = max(abs(fd[row]), abs(gradients[row, e]), 1e-12)
            assert abs(fd[row] - gradients[row, e]) / scale < 1e-3


def test_formulation_swap_changes_the_constraint_set(
    mesh, element, dofs, assembler, selection, design
):
    """Eq. (15) has three constraints; Eq. (21) folds two of them in and keeps one."""
    constrained = make_analysis(
        mesh, element, dofs, assembler, selection, formulation=ConstraintFormulation()
    )
    multi = make_analysis(
        mesh, element, dofs, assembler, selection, formulation=MultiObjectiveFormulation()
    )
    _, r1, g1 = constrained.evaluate(design)
    _, r2, g2 = multi.evaluate(design)

    assert r1.n_responses == 4 and g1.shape[0] == 4
    assert r2.n_responses == 2 and g2.shape[0] == 2
    assert r1.names[1:] == ("avg_shape", "var_surround", "volume")
    assert r2.names[1:] == ("volume",)


def test_element_swap_changes_the_stiffness_not_the_pipeline(mesh, dofs):
    """A different shear scheme is a different element, same interface."""
    from chladni_topopt.assembly import Assembler

    mitc = Assembler(mesh, MindlinQ4(thickness=0.2, shear_scheme="mitc4"), dofs)
    full = Assembler(mesh, MindlinQ4(thickness=0.2, shear_scheme="full"), dofs)
    x = np.full(mesh.n_elem, 1.0)

    # Full integration locks, so it must be stiffer than MITC4.
    assert abs(full.stiffness(x)).sum() > abs(mitc.stiffness(x)).sum()
    assert np.allclose(full.me0, mitc.me0)


class ConstantStepOptimizer:
    """A trivial Optimizer, to show the driver only talks to the protocol."""

    def __init__(self, step=0.01):
        self.step = step
        self.calls = 0

    def update(self, x, f0, df0, fi, dfi):
        self.calls += 1
        return np.clip(x - self.step * np.sign(df0), 0.0, 1.0)


def test_a_foreign_optimizer_drives_the_loop(analysis, mesh):
    optimizer = ConstantStepOptimizer(step=0.02)
    assert isinstance(optimizer, Optimizer)

    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=optimizer,
        continuation=BetaContinuation(
            projection=analysis.projection, beta0=1.0, beta_max=1.0
        ),
        convergence=WindowedConvergence(window=2, tol=0.002),
        max_iterations=4,
        verbose=False,
    )
    result = driver.run(np.full(mesh.n_elem, 0.5))
    assert result.iterations == 4
    assert optimizer.calls == 4
    assert np.all((result.x >= 0.0) & (result.x <= 1.0))


def test_identity_filter_isolates_the_projection(mesh, element, dofs, assembler, selection, design):
    analysis = make_analysis(
        mesh, element, dofs, assembler, selection, density_filter=IdentityFilter()
    )
    state = analysis.solve(design)
    assert np.allclose(state.design.xtilde, design)
    assert np.allclose(
        state.design.xbar, ExponentialThreshold(eta=0.5, beta=1.0).forward(design)
    )


def test_boundary_condition_swap(mesh):
    from chladni_topopt.bc import ClampedEdges, CornerClamps, SimplySupportedEdges

    counts = {
        type(bc).__name__: len(DofSetFor(mesh, bc))
        for bc in (CornerClamps(0.1), ClampedEdges(), SimplySupportedEdges())
    }
    assert counts["SimplySupportedEdges"] * 3 == counts["ClampedEdges"]
    assert counts["CornerClamps"] < counts["ClampedEdges"]


def DofSetFor(mesh, bc):
    from chladni_topopt.bc import DofSet

    return DofSet.from_mesh(mesh, bc).fixed
