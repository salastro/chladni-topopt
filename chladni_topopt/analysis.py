"""The forward problem: design variables -> modal state -> responses and gradients.

This is the whole analysis chain in one object, deliberately separate from the
optimization loop in :mod:`chladni_topopt.driver`.  Keeping them apart is what
lets the sensitivity test drive the same code the optimizer uses, with nothing
stubbed out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .assembly import Assembler
from .bc import DofSet
from .eigen import EigenSolution, EigenSolver
from .filters import Filter, Projection
from .formulations import Formulation
from .linalg import Factorizer, default_factorizer
from .material import MaterialInterpolation
from .selection import SelectionVectors
from .sensitivity import AdjointEigenvectorSensitivity
from .state import DesignState, ReferenceIndicators, Responses, TargetMode
from .tracking import ModeTracker, TrackResult


@dataclass(frozen=True)
class ModalState:
    """Everything one forward solve produces."""

    design: DesignState
    stiffness: sp.spmatrix
    mass: sp.spmatrix
    solution: EigenSolution
    modes: tuple[TargetMode, ...]
    tracking: tuple[TrackResult, ...]


@dataclass
class ChladniAnalysis:
    """Assembles, solves the eigenproblem, tracks modes and evaluates responses.

    The components are all injected: swap ``element``, ``material``, ``filter``,
    ``projection``, ``eigen_solver``, ``trackers`` or ``formulation`` without
    touching this class.
    """

    assembler: Assembler
    dofs: DofSet
    material: MaterialInterpolation
    filter: Filter
    projection: Projection
    eigen_solver: EigenSolver
    formulation: Formulation
    selections: Sequence[SelectionVectors]
    trackers: Sequence[ModeTracker]
    weights: Sequence[float] = (1.0,)
    element_volumes: np.ndarray = field(default=None, repr=False)
    #: Factorization backend for the bordered adjoint solve.  ``build_problem``
    #: passes the same one to the eigensolver.
    factorizer: Factorizer = field(default_factory=default_factorizer)
    references: tuple[ReferenceIndicators, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.element_volumes is None:
            # Element areas, not volumes: the plate has uniform thickness and
            # the volume fraction normalises by the total, so any constant
            # factor cancels.  Reading a thickness off the element would also
            # reach past the ElementFormulation protocol.
            mesh = self.assembler.mesh
            self.element_volumes = np.full(mesh.n_elem, mesh.elem_area)
        if len(self.selections) != len(self.trackers):
            raise ValueError("need one mode tracker per target pattern")
        self.sensitivity = AdjointEigenvectorSensitivity(
            assembler=self.assembler,
            dofs=self.dofs,
            material=self.material,
            filter=self.filter,
            projection=self.projection,
            factorizer=self.factorizer,
        )

    # -- forward ---------------------------------------------------------

    def design_state(self, x: np.ndarray) -> DesignState:
        xtilde = self.filter.forward(x)
        xbar = self.projection.forward(xtilde)
        return DesignState(
            x=x, xtilde=xtilde, xbar=xbar, volumes=self.element_volumes
        )

    def solve(self, x: np.ndarray) -> ModalState:
        """Assemble, solve the eigenproblem and identify the target modes."""
        design = self.design_state(x)
        k = self.assembler.stiffness(self.material.youngs(design.xbar))
        m = self.assembler.mass(self.material.density(design.xbar))
        solution = self.eigen_solver.solve(k, m, self.dofs)

        if not self.references:
            self.references = tuple(
                ReferenceIndicators.from_mode(
                    sel, solution.full[:, tracker.index]
                )
                for sel, tracker in zip(self.selections, self.trackers)
            )

        modes, tracking = [], []
        for sel, tracker, ref, weight in zip(
            self.selections, self.trackers, self.references, self.weights
        ):
            result = tracker.select(solution)
            lam, phi_red, phi_full = solution.mode(result.index)
            modes.append(
                TargetMode(
                    index=result.index,
                    selection=sel,
                    reference=ref,
                    weight=weight,
                    value=lam,
                    phi_full=phi_full,
                    phi_reduced=phi_red,
                )
            )
            tracking.append(result)

        return ModalState(
            design=design,
            stiffness=k,
            mass=m,
            solution=solution,
            modes=tuple(modes),
            tracking=tuple(tracking),
        )

    def responses(self, state: ModalState) -> Responses:
        return self.formulation.responses(state.modes, state.design)

    def gradients(self, state: ModalState, responses: Responses) -> np.ndarray:
        return self.sensitivity.gradients(
            responses, state.modes, state.design, state.stiffness, state.mass
        )

    def evaluate(self, x: np.ndarray) -> tuple[ModalState, Responses, np.ndarray]:
        """One full analysis: state, responses, gradients."""
        state = self.solve(x)
        responses = self.responses(state)
        return state, responses, self.gradients(state, responses)
