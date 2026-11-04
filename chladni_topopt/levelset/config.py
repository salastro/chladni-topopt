"""Configuration and factory for the level-set (boundary-geometry) problem.

The plate, supports, targets, formulation and solvers come from the same
:class:`~chladni_topopt.config.ProblemConfig` as the density method; only the
design parameterisation differs, and it is described by :class:`LevelSetConfig`.
:func:`build_levelset_problem` swaps four components of the density chain::

    density method                 level-set method
    x (element densities)    ->    Phi (nodal level set)
    DensityFilter            ->    NodalToElement
    ExponentialThreshold     ->    WangHeaviside
    SIMPTwoMaterial          ->    SolidVoidErsatz
    BetaContinuation         ->    NoContinuation
    AdamW / MMA              ->    LevelSetUpdate(HamiltonJacobi | AdamW)

and leaves the analysis, adjoint, mode tracking and driver as they are.  Two
additions keep the design usable as a Chladni plate: a solid corridor along
the target, so no hole can sit under the nodal line, and an optional
perimeter penalty that prefers few, large holes over many small ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..analysis import ChladniAnalysis
from ..config import Problem, ProblemConfig, build_common
from ..domain import PlateInDomain
from ..driver import TopologyOptimizer, WindowedConvergence
from ..eigen import ShiftInvertEigen
from ..filters import NoContinuation
from ..material import SolidVoidErsatz
from ..optimizers.adamw import AdamW
from ..optimizers.hamilton_jacobi import HamiltonJacobi
from ..regularizers import PerimeterTV, RegularizedFormulation
from .heaviside import WangHeaviside
from .initial import frozen_nodes, live_edge, perforated, target_corridor
from .mapping import NodalToElement
from .reinit import ContourReinitializer
from .update import LevelSetUpdate


@dataclass(frozen=True)
class LevelSetConfig:
    """Level-set parameterisation, update and initial design.

    Lengths are in metres, so a quick preview and a full-resolution run solve
    the same physical problem (in element widths, halving the element halved
    the Heaviside width and the step too).  The defaults are the values tuned
    on the 0.04 m quick mesh.  The plate material is
    ``ProblemConfig.material.e1`` and ``rho1``; the void is an ersatz fraction
    of it.  The volume cap is the formulation's ``vstar``, the fraction of the
    domain left solid; ``ConstraintFormulation(vstar=None)`` drops it.

    With ``domain`` set, the mesh is that design domain and the plate starts
    as the rectangle inside it, so its outline is a zero contour from the
    first iteration and can move outward as well as inward.
    """

    #: ``"hj"``: Wang et al.'s upwind Hamilton-Jacobi step.  ``"adamw"``: AdamW
    #: on the nodal values of ``Phi``, a parametric level set.
    update: Literal["hj", "adamw"] = "hj"
    #: Half-width of the smoothed Heaviside, Eq. (40), in metres.  Wang et al.
    #: use two elements; this is two quick-mesh elements.
    width: float = 0.08
    e_void: float = 1e-3
    rho_void: float = 1e-5
    #: Initial perforation: holes along x and y, and their radius as a
    #: fraction of the hole spacing (see :func:`.initial.perforated`).
    holes: tuple[int, int] = (4, 5)
    hole_radius: float = 0.35
    reinit_every: int = 5
    #: Keep the target's band and gap zones (plus a Heaviside width) solid.
    protect_target: bool = True
    #: Weight of ``P / P0`` added to the objective; 0 turns the penalty off.
    perimeter_weight: float = 0.2
    #: The plate inside a larger design domain (see :mod:`chladni_topopt.domain`);
    #: ``None`` means the mesh is the plate.
    domain: PlateInDomain | None = None
    #: Without a domain: make the plate outline a moving boundary, starting
    #: ``edge_margin`` metres inside the edge (see :func:`.initial.live_edge`).
    #: It can only move inward.
    live_edge: bool = False
    edge_margin: float = 0.02

    # Hamilton-Jacobi: boundary move per update, in metres, decaying
    # geometrically to ``step_min``.
    step: float = 0.008
    step_decay: float = 0.996
    step_min: float = 0.002
    #: Length over which the speed is smoothed (Wang's non-local ``mu``); the
    #: number of averaging passes scales with ``(smooth_length / h)^2``.
    smooth_length: float = 0.057

    # AdamW on Phi: step sizes and the bound on |Phi|, in metres.
    lr: float = 0.02
    lr_min: float = 0.002
    bound: float = 0.24


def build_levelset_problem(
    config: ProblemConfig, levelset: LevelSetConfig = LevelSetConfig(), verbose: bool = True
) -> Problem:
    """Assemble the level-set problem; ``config.optimizer`` supplies the
    iteration budget, convergence test and augmented-Lagrangian settings."""
    parts = build_common(config, verbose)
    mesh, unit = parts.mesh, parts.unit
    width = levelset.width
    h = min(mesh.dx, mesh.dy)
    opt = config.optimizer
    m = len(config.formulation.constraint_names)
    formulation = config.formulation
    if levelset.perimeter_weight > 0.0:
        formulation = RegularizedFormulation(
            inner=formulation, perimeter=PerimeterTV(mesh), weight=levelset.perimeter_weight
        )

    analysis = ChladniAnalysis(
        assembler=parts.assembler,
        dofs=parts.dofs,
        material=SolidVoidErsatz(
            e=config.material.e1,
            rho=config.material.rho1,
            e_void=levelset.e_void,
            rho_void=levelset.rho_void,
        ),
        filter=NodalToElement(mesh),
        projection=WangHeaviside(width=width),
        eigen_solver=ShiftInvertEigen(n_modes=parts.n_modes, factorizer=parts.factorizer),
        factorizer=parts.factorizer,
        formulation=formulation,
        selections=parts.selections,
        trackers=parts.trackers,
        weights=parts.weights,
    )

    if levelset.update == "hj":
        inner = HamiltonJacobi(
            grid=(mesh.nx + 1, mesh.ny + 1),
            dx=mesh.dx,
            dy=mesh.dy,
            m=m,
            cfl=levelset.step / h,
            cfl_decay=levelset.step_decay,
            cfl_min=levelset.step_min / h,
            # n passes of 5-point averaging spread over about h * sqrt(n / 2).
            smooth_passes=round(2.0 * (levelset.smooth_length / h) ** 2),
            rho=opt.rho,
            dual_interval=opt.dual_interval,
        )
    elif levelset.update == "adamw":
        inner = AdamW(
            n=mesh.n_node,
            m=m,
            xmin=-levelset.bound,
            xmax=levelset.bound,
            lr=levelset.lr,
            lr_decay=opt.lr_decay,
            lr_min=levelset.lr_min,
            beta1=opt.beta1,
            beta2=opt.beta2,
            rho=opt.rho,
            dual_interval=opt.dual_interval,
        )
    else:
        raise ValueError(f"unknown level-set update {levelset.update!r}")

    # The clamps, and a Heaviside width around them, stay solid -- and so does
    # the zone each target's nodal line may occupy, plus a Heaviside width so
    # the elements at the corridor's edge are fully solid too.
    frozen = [frozen_nodes(mesh, parts.dofs, margin=width)]
    if levelset.protect_target:
        for target, selection in zip(config.targets, parts.selections):
            half_width = (target.band_elements + target.gap_elements) * unit + width
            frozen.append(target_corridor(mesh, selection, half_width))
    frozen = np.unique(np.concatenate(frozen))
    update = LevelSetUpdate(
        inner=inner,
        reinit=ContourReinitializer(mesh),
        frozen=frozen,
        frozen_value=width,
        reinit_every=levelset.reinit_every,
    )
    if levelset.domain is not None:
        domain = levelset.domain
        phi0 = perforated(
            mesh, levelset.holes, levelset.hole_radius,
            box=(*domain.origin, *domain.plate),
        )
        phi0 = np.minimum(phi0, domain.plate_sdf(mesh.node_coords()))
    else:
        phi0 = perforated(mesh, levelset.holes, levelset.hole_radius)
        if levelset.live_edge:
            phi0 = live_edge(mesh, phi0, levelset.edge_margin)
    phi0 = update.freeze(phi0)

    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=update,
        continuation=NoContinuation(),
        convergence=WindowedConvergence(window=opt.window, tol=opt.tol),
        max_iterations=opt.max_iterations,
        verbose=verbose,
    )

    return Problem(
        config=config,
        mesh=mesh,
        dofs=parts.dofs,
        assembler=parts.assembler,
        analysis=analysis,
        driver=driver,
        initial=phi0,
    )
