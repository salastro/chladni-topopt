"""Adjoint sensitivity of eigenvector-dependent responses, Eqs. (16)-(18).

For a response ``C(phi, lambda)`` constrained by the eigenproblem
``R1 = (K - lambda M) phi = 0`` and the normalisation
``R2 = (phi^T M phi - 1) / 2 = 0``, stationarity of the Lagrangian
``C + u^T R1 + v R2`` with respect to ``phi`` and ``lambda`` gives the bordered
adjoint system::

    [ K - lambda M    M phi ] [u]   [ -dC/dphi ]
    [     phi^T M       0   ] [v] = [  dC/dlam ]

after which

    dC/dxbar_e = u^T (dK/dxbar_e - lambda dM/dxbar_e) phi
               + (v/2) phi^T (dM/dxbar_e) phi

This is Eqs. (16)-(17) up to the paper's factor-of-two scaling of ``u`` and
``v``; the convention here is fixed by :mod:`tests.test_sensitivity`, which
checks every response against central differences.

``K - lambda M`` on its own is singular -- ``phi`` spans its null space -- so
the bordering is not a convenience, it is what makes the system solvable.  The
bordered matrix is the same for every response of a given mode, so it is
factorised once per mode per iteration and back-substituted for each response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import scipy.sparse as sp

from .assembly import Assembler
from .bc import DofSet
from .filters import Filter, Projection
from .linalg import Factorizer, default_factorizer
from .material import MaterialInterpolation
from .state import DesignState, Responses, TargetMode


@runtime_checkable
class SensitivityAnalysis(Protocol):
    """Maps response partials to gradients with respect to design variables."""

    def gradients(
        self,
        responses: Responses,
        modes: Sequence[TargetMode],
        design: DesignState,
        k: sp.spmatrix,
        m: sp.spmatrix,
    ) -> np.ndarray:
        """``(n_responses, n_design)`` array of ``dC/dx``."""
        ...


@dataclass
class AdjointEigenvectorSensitivity:
    """Bordered-adjoint sensitivities chained back through filter and projection."""

    assembler: Assembler
    dofs: DofSet
    material: MaterialInterpolation
    filter: Filter
    projection: Projection
    factorizer: Factorizer = field(default_factory=default_factorizer)

    def _bordered(
        self, k: sp.spmatrix, m: sp.spmatrix, lam: float, phi_red: np.ndarray
    ):
        m_phi = np.asarray(m @ phi_red).reshape(-1, 1)
        border = sp.csc_matrix(m_phi)
        matrix = sp.bmat(
            [
                [(k - lam * m).tocsc(), border],
                [border.T, sp.csc_matrix((1, 1))],
            ],
            format="csc",
        )
        # Symmetric but indefinite: it is a saddle-point system, and K - lam M
        # already has negative eigenvalues below the target mode.
        return self.factorizer.factorize(matrix, definite=False)

    def dc_dxbar(
        self,
        responses: Responses,
        modes: Sequence[TargetMode],
        design: DesignState,
        k: sp.spmatrix,
        m: sp.spmatrix,
    ) -> np.ndarray:
        """``(n_responses, n_elem)`` derivative with respect to physical density."""
        n_resp = responses.n_responses
        total = responses.dxbar.copy()

        d_youngs = self.material.d_youngs(design.xbar)
        d_density = self.material.d_density(design.xbar)
        free = self.dofs.free

        for s, mode in enumerate(modes):
            active = [
                r
                for r in range(n_resp)
                if np.any(responses.dphi[r, s]) or responses.dlam[r, s] != 0.0
            ]
            if not active:
                continue

            lu = self._bordered(k, m, mode.value, mode.phi_reduced)
            # phi^T (dM/dxbar_e) phi is shared by every response of this mode.
            q_mass_phi = self.assembler.element_quadratic(
                mode.phi_full, mode.phi_full, "m"
            )

            # Every active response of this mode in one multi-RHS solve.
            rhs = np.empty((self.dofs.n_free + 1, len(active)))
            for col, r in enumerate(active):
                rhs[:-1, col] = -responses.dphi[r, s][free]
                rhs[-1, col] = responses.dlam[r, s]
            sols = lu.solve(rhs)

            for col, r in enumerate(active):
                u_full = self.dofs.expand(sols[:-1, col])
                v = sols[-1, col]

                q_k = self.assembler.element_quadratic(u_full, mode.phi_full, "k")
                q_m = self.assembler.element_quadratic(u_full, mode.phi_full, "m")

                total[r] += (
                    d_youngs * q_k
                    - mode.value * d_density * q_m
                    + 0.5 * v * d_density * q_mass_phi
                )

        return total

    def gradients(
        self,
        responses: Responses,
        modes: Sequence[TargetMode],
        design: DesignState,
        k: sp.spmatrix,
        m: sp.spmatrix,
    ) -> np.ndarray:
        """``(n_responses, n_design)`` gradients, Eq. (18)."""
        dxbar = self.dc_dxbar(responses, modes, design, k, m)
        dproj = self.projection.gradient(design.xtilde)
        return np.vstack([self.filter.backward(dproj * row) for row in dxbar])
