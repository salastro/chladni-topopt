"""Configuration and factory for the spline-outline problem.

The plate is the region inside one closed B-spline, inside a void-filled
design domain (:class:`~chladni_topopt.domain.PlateInDomain`).  There are no
holes: the design is the outline alone, a few dozen offsets.  The chain is the
level-set method's with the nodal map swapped for the spline::

    level-set method               spline method
    x = nodal Phi            ->    x = control-point offsets u in [0, 1]^n
    NodalToElement           ->    SplineMap (element signed distance)
    WangHeaviside, SolidVoidErsatz, NoContinuation: unchanged
    LevelSetUpdate(HJ)       ->    MMA or AdamW from OptimizerConfig

With so few bounded variables MMA is the natural optimizer, so the factory
uses ``ProblemConfig.optimizer`` as it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis import ChladniAnalysis
from ..config import Problem, ProblemConfig, build_common
from ..domain import PlateInDomain
from ..driver import TopologyOptimizer, WindowedConvergence
from ..eigen import ShiftInvertEigen
from ..filters import NoContinuation
from ..levelset.heaviside import WangHeaviside
from ..material import SolidVoidErsatz
from ..regularizers import PerimeterTV, RegularizedFormulation
from ..selection import signed_distance_to_target
from .geometry import ClosedBSpline, rectangle_controls
from .mapping import SplineMap


@dataclass(frozen=True)
class SplineConfig:
    """The outline's parameterisation.  Lengths in metres."""

    domain: PlateInDomain = PlateInDomain()
    n_ctrl: int = 48
    #: How far a control point may move inward from the starting rectangle.
    #: Outward it may move to the domain edge (less ``edge_clearance``).
    t_in: float = 0.6
    edge_clearance: float = 0.04
    #: Half-width of the smoothed Heaviside on the signed distance.
    width: float = 0.08
    e_void: float = 1e-3
    rho_void: float = 1e-5
    #: Keep the target's band and gap zones (plus a Heaviside width) solid.
    protect_target: bool = True
    perimeter_weight: float = 0.2
    samples: int = 2000


def build_spline_problem(
    config: ProblemConfig, spline: SplineConfig = SplineConfig(), verbose: bool = True
) -> Problem:
    """Assemble the spline-outline problem.

    ``config.mesh`` and ``config.boundary`` must describe ``spline.domain``
    (``domain.mesh()`` and ``domain.clamps()``), and the targets must already
    be in domain coordinates (``domain.shift``).
    """
    parts = build_common(config, verbose)
    mesh, unit, domain = parts.mesh, parts.unit, spline.domain
    centroids = mesh.elem_centroids()

    controls, directions = rectangle_controls(*domain.origin, *domain.plate, spline.n_ctrl)
    t_hi = max(domain.margin - spline.edge_clearance, 0.0)
    t_lo = -spline.t_in

    # Elements that must stay solid: anything touching a clamped node, and the
    # target corridor (plus a Heaviside width so its edge elements are solid).
    fixed_node = np.zeros(mesh.n_node, dtype=bool)
    fixed_node[np.unique(parts.dofs.fixed // mesh.ndof_per_node)] = True
    frozen = [np.flatnonzero(fixed_node[mesh.connectivity()].any(axis=1))]
    if spline.protect_target:
        for target in config.targets:
            reach = (target.band_elements + target.gap_elements) * unit + spline.width
            d = signed_distance_to_target(centroids, target.loops, target.open_curves)
            frozen.append(np.flatnonzero(np.abs(d) <= reach))
    frozen = np.unique(np.concatenate(frozen))

    spline_map = SplineMap(
        spline=ClosedBSpline(n_ctrl=len(controls), samples=spline.samples),
        controls=controls,
        directions=directions,
        t_lo=t_lo,
        t_hi=t_hi,
        centroids=centroids,
        band=spline.width,
        frozen=frozen,
        frozen_value=spline.width,
    )

    formulation = config.formulation
    if spline.perimeter_weight > 0.0:
        formulation = RegularizedFormulation(
            inner=formulation, perimeter=PerimeterTV(mesh), weight=spline.perimeter_weight
        )

    analysis = ChladniAnalysis(
        assembler=parts.assembler,
        dofs=parts.dofs,
        material=SolidVoidErsatz(
            e=config.material.e1,
            rho=config.material.rho1,
            e_void=spline.e_void,
            rho_void=spline.rho_void,
        ),
        filter=spline_map,
        projection=WangHeaviside(width=spline.width),
        eigen_solver=ShiftInvertEigen(n_modes=parts.n_modes, factorizer=parts.factorizer),
        factorizer=parts.factorizer,
        formulation=formulation,
        selections=parts.selections,
        trackers=parts.trackers,
        weights=parts.weights,
    )

    opt = config.optimizer
    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=opt.build(n=spline_map.n, m=len(config.formulation.constraint_names)),
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
        initial=spline_map.u_of(np.zeros(spline_map.n)),
    )
