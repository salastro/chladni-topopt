"""Configuration dataclasses and a factory that wires the components together.

:func:`build_problem` is a convenience, not a bottleneck: every component it
constructs can equally be built by hand and passed to
:class:`~chladni_topopt.analysis.ChladniAnalysis` and
:class:`~chladni_topopt.driver.TopologyOptimizer` directly.  Swapping a
component means passing a different object, not editing this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Sequence

import numpy as np

from .analysis import ChladniAnalysis
from .assembly import Assembler
from .bc import BoundaryCondition, CornerClamps, DofSet
from .driver import TopologyOptimizer, WindowedConvergence
from .eigen import ShiftInvertEigen
from .elements.mindlin_q4 import MindlinQ4
from .filters import BetaContinuation, DensityFilter, ExponentialThreshold
from .formulations import ConstraintFormulation, Formulation
from .linalg import make_factorizer
from .material import SIMPTwoMaterial
from .mesh import StructuredQuadMesh
from .optimizers.adamw import AdamW
from .optimizers.base import Optimizer
from .optimizers.mma import MMA
from .selection import build_selection_vectors
from .tracking import MACTracker


@dataclass(frozen=True)
class MeshConfig:
    """Plate geometry and discretisation."""

    lx: float = 2.0
    ly: float = 2.4
    nx: int = 100
    ny: int = 120
    thickness: float = 0.2

    def build(self) -> StructuredQuadMesh:
        return StructuredQuadMesh(self.lx, self.ly, self.nx, self.ny)


@dataclass(frozen=True)
class MaterialConfig:
    """Two-material SIMP properties.  Defaults are the paper's aluminium/epoxy."""

    e1: float = 68.5e9
    e2: float = 1.0e9
    rho1: float = 2.7e3
    rho2: float = 1.18e3
    penal: float = 3.0
    poisson: float = 0.3
    shear_correction: float = 5.0 / 6.0

    def build(self) -> SIMPTwoMaterial:
        return SIMPTwoMaterial(
            e1=self.e1, e2=self.e2, rho1=self.rho1, rho2=self.rho2, penal=self.penal
        )


@dataclass(frozen=True)
class FilterConfig:
    """Density filter radius and threshold-projection continuation."""

    #: Filter radius in element widths; the paper uses 4.
    rmin_elements: float = 4.0
    eta: float = 0.5
    beta0: float = 1.0
    beta_max: float = 256.0
    beta_interval: int = 50


@dataclass(frozen=True)
class TargetConfig:
    """One target Chladni pattern: the curve, its zones, and which mode carries it.

    ``loops`` are closed curves and ``open_curves`` are curves that end.  Pass
    ``loops=()`` for a target made only of open curves.  A nodal line cannot
    stop in mid-plate, so an open curve should run out to the plate edges.
    """

    loops: Sequence[np.ndarray]
    mode_index: int
    weight: float = 1.0
    #: Zone half-widths, in element widths (see Fig. 2b).
    #:
    #: ``band_elements`` is the one that matters most, and not for the reason
    #: Fig. 2(b) suggests.  How small ``Avg(L, phi)`` can get scales with the
    #: square of the band width -- see
    #: :func:`~chladni_topopt.selection.linear_profile_shape_ratio` -- and at
    #: two elements the estimate is about 0.042, four times the paper's
    #: ``eps = 0.01``.  One element brings it under 0.01.
    band_elements: float = 1.0
    gap_elements: float = 2.0
    surround_elements: float = 8.0
    open_curves: Sequence[np.ndarray] = ()

    def __post_init__(self) -> None:
        if not len(self.loops) and not len(self.open_curves):
            raise ValueError("a target needs at least one loop or open curve")


@dataclass(frozen=True)
class OptimizerConfig:
    """Which design-update method to use, its settings, and the iteration budget.

    ``method`` picks the optimizer.  Only the fields of the chosen method are
    read; the others are ignored.
    """

    method: Literal["adamw", "mma"] = "adamw"
    max_iterations: int = 600
    tol: float = 0.002
    window: int = 5

    # AdamW (augmented Lagrangian, projected onto the box).
    lr: float = 0.02
    lr_decay: float = 0.9962
    lr_min: float = 0.002
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    decay_target: float = 0.0
    rho: float = 10.0
    dual_interval: int = 5

    # MMA.
    move: float = 0.1
    asyinit: float = 0.5
    asyincr: float = 1.2
    asydecr: float = 0.7

    def build(self, n: int, m: int) -> Optimizer:
        if self.method == "adamw":
            return AdamW(
                n=n,
                m=m,
                lr=self.lr,
                lr_decay=self.lr_decay,
                lr_min=self.lr_min,
                beta1=self.beta1,
                beta2=self.beta2,
                weight_decay=self.weight_decay,
                decay_target=self.decay_target,
                rho=self.rho,
                dual_interval=self.dual_interval,
            )
        if self.method == "mma":
            return MMA(
                n=n,
                m=m,
                move=self.move,
                asyinit=self.asyinit,
                asyincr=self.asyincr,
                asydecr=self.asydecr,
            )
        raise ValueError(f"unknown optimizer method {self.method!r}")


@dataclass(frozen=True)
class ProblemConfig:
    """Everything needed to define a run."""

    targets: Sequence[TargetConfig]
    mesh: MeshConfig = field(default_factory=MeshConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    formulation: Formulation = field(default_factory=ConstraintFormulation)
    boundary: BoundaryCondition = field(default_factory=CornerClamps)
    #: Lowest modes computed each iteration.  ``build_problem`` raises this to
    #: at least ``target mode index + MODE_HEADROOM`` so tracking can follow
    #: the target even if a few modes cross it.
    n_modes: int = 8
    #: Sparse factorization backend: "auto" picks PARDISO when ``pypardiso`` is
    #: installed and falls back to SuperLU.
    solver: Literal["auto", "superlu", "pardiso"] = "auto"
    x0: float = 0.5

    def coarsen(self, factor: int = 2) -> "ProblemConfig":
        """The same *physical* problem on a coarser mesh.

        Used by the ``--quick`` preset and the test suite.  Radii and zone
        widths are stored in element widths, so they must be divided by the
        same factor the mesh is divided by -- otherwise coarsening silently
        inflates the filter radius and the selection bands, and the cheap run
        stops being a preview of the expensive one.

        Widths are floored at one element: below that a zone falls between
        nodes and selects nothing.
        """
        mesh = replace(
            self.mesh, nx=self.mesh.nx // factor, ny=self.mesh.ny // factor
        )
        filt = replace(
            self.filter, rmin_elements=max(self.filter.rmin_elements / factor, 1.0)
        )
        targets = [
            replace(
                t,
                band_elements=max(t.band_elements / factor, 1.0),
                gap_elements=max(t.gap_elements / factor, 1.0),
                surround_elements=max(t.surround_elements / factor, 1.0),
            )
            for t in self.targets
        ]
        return replace(self, mesh=mesh, filter=filt, targets=targets)


@dataclass
class Problem:
    """Constructed components, kept accessible so callers can inspect or replace them."""

    config: ProblemConfig
    mesh: StructuredQuadMesh
    dofs: DofSet
    assembler: Assembler
    analysis: ChladniAnalysis
    driver: TopologyOptimizer
    #: Starting design.  ``None`` means the uniform density ``config.x0``; a
    #: level-set problem stores its initial nodal ``Phi`` here.
    initial: np.ndarray | None = None

    @property
    def x0(self) -> np.ndarray:
        if self.initial is not None:
            return self.initial.copy()
        return np.full(self.mesh.n_elem, self.config.x0)

    def run(self):
        return self.driver.run(self.x0)


#: Modes computed above the highest target, so MAC tracking has room to
#: follow a target whose position in the spectrum shifts during the run.
MODE_HEADROOM = 4


@dataclass
class CommonParts:
    """The components every design parameterisation shares.

    Density and level-set problems differ only in how the design variables
    become physical densities, how they are updated, and the material model;
    the plate, its supports, the targets and the solvers are the same.
    """

    factorizer: object
    n_modes: int
    mesh: StructuredQuadMesh
    element: MindlinQ4
    dofs: DofSet
    assembler: Assembler
    selections: list
    trackers: list
    weights: list

    @property
    def unit(self) -> float:
        """Element width used to convert ``*_elements`` settings to metres."""
        return 0.5 * (self.mesh.dx + self.mesh.dy)


def build_common(config: ProblemConfig, verbose: bool = True) -> CommonParts:
    """Mesh, element, supports, assembler, targets and solver backend."""
    factorizer = make_factorizer(config.solver)
    n_modes = max(
        config.n_modes, max(t.mode_index for t in config.targets) + MODE_HEADROOM
    )
    if verbose:
        print(f"sparse solver: {factorizer.name}   modes computed: {n_modes}")
    mesh = config.mesh.build()
    element = MindlinQ4(
        thickness=config.mesh.thickness,
        poisson=config.material.poisson,
        shear_correction=config.material.shear_correction,
    )
    dofs = DofSet.from_mesh(mesh, config.boundary)
    assembler = Assembler(mesh, element, dofs)
    unit = 0.5 * (mesh.dx + mesh.dy)

    selections, trackers, weights = [], [], []
    for target in config.targets:
        selections.append(
            build_selection_vectors(
                mesh,
                dofs,
                target.loops,
                band=target.band_elements * unit,
                gap=target.gap_elements * unit,
                surround=target.surround_elements * unit,
                transverse_dof=element.transverse_dof,
                open_curves=target.open_curves,
            )
        )
        trackers.append(MACTracker(index=target.mode_index))
        weights.append(target.weight)

    return CommonParts(
        factorizer=factorizer,
        n_modes=n_modes,
        mesh=mesh,
        element=element,
        dofs=dofs,
        assembler=assembler,
        selections=selections,
        trackers=trackers,
        weights=weights,
    )


def build_problem(config: ProblemConfig, verbose: bool = True) -> Problem:
    """Assemble every component of the density (SIMP) problem from a configuration."""
    parts = build_common(config, verbose)
    mesh = parts.mesh
    density_filter = DensityFilter(
        rmin=config.filter.rmin_elements * parts.unit, centroids=mesh.elem_centroids()
    )
    projection = ExponentialThreshold(eta=config.filter.eta, beta=config.filter.beta0)

    analysis = ChladniAnalysis(
        assembler=parts.assembler,
        dofs=parts.dofs,
        material=config.material.build(),
        filter=density_filter,
        projection=projection,
        eigen_solver=ShiftInvertEigen(n_modes=parts.n_modes, factorizer=parts.factorizer),
        factorizer=parts.factorizer,
        formulation=config.formulation,
        selections=parts.selections,
        trackers=parts.trackers,
        weights=parts.weights,
    )

    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=config.optimizer.build(
            n=mesh.n_elem, m=len(config.formulation.constraint_names)
        ),
        continuation=BetaContinuation(
            projection=projection,
            beta0=config.filter.beta0,
            beta_max=config.filter.beta_max,
            interval=config.filter.beta_interval,
        ),
        convergence=WindowedConvergence(
            window=config.optimizer.window, tol=config.optimizer.tol
        ),
        max_iterations=config.optimizer.max_iterations,
        verbose=verbose,
    )

    return Problem(
        config=config,
        mesh=mesh,
        dofs=parts.dofs,
        assembler=parts.assembler,
        analysis=analysis,
        driver=driver,
    )
