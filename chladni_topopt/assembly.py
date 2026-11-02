"""Global matrix assembly.

Eqs. (8)-(9) factor the material properties out of the element matrices::

    K = sum_e E_e k_e        M = sum_e rho_e m_e

On a uniform mesh every element shares the same ``k_e`` and ``m_e``, so they
are formed once and each assembly is a scatter-add of rescaled copies.  The
triplet indices are precomputed too, and the fixed DOFs are dropped at the
triplet stage rather than by slicing the assembled matrix -- that turns out to
dominate the cost otherwise, and this runs several hundred times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .bc import DofSet
from .elements.base import ElementFormulation
from .mesh import StructuredQuadMesh


@dataclass
class Assembler:
    """Builds reduced ``K`` and ``M`` from per-element property arrays."""

    mesh: StructuredQuadMesh
    element: ElementFormulation
    dofs: DofSet

    ke0: np.ndarray = field(init=False)
    me0: np.ndarray = field(init=False)
    element_dofs: np.ndarray = field(init=False)
    _rows: np.ndarray = field(init=False, repr=False)
    _cols: np.ndarray = field(init=False, repr=False)
    _ke_flat: np.ndarray = field(init=False, repr=False)
    _me_flat: np.ndarray = field(init=False, repr=False)
    _elem_of_entry: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        coords = self.mesh.node_coords()
        conn = self.mesh.connectivity()
        self.ke0, self.me0 = self.element.unit_matrices(coords[conn[0]])

        self.element_dofs = self.mesh.element_dofs()
        n_elem, n_edof = self.element_dofs.shape

        # Full triplet lists, then drop anything touching a fixed DOF.
        rows = np.repeat(self.element_dofs, n_edof, axis=1).ravel()
        cols = np.tile(self.element_dofs, (1, n_edof)).ravel()

        red = self.dofs.reduced_index()
        r_red, c_red = red[rows], red[cols]
        keep = (r_red >= 0) & (c_red >= 0)

        self._rows = r_red[keep]
        self._cols = c_red[keep]
        self._elem_of_entry = np.repeat(np.arange(n_elem), n_edof * n_edof)[keep]
        self._ke_flat = np.tile(self.ke0.ravel(), n_elem)[keep]
        self._me_flat = np.tile(self.me0.ravel(), n_elem)[keep]

    @property
    def n_free(self) -> int:
        return self.dofs.n_free

    def _assemble(self, scale: np.ndarray, flat: np.ndarray) -> sp.csc_matrix:
        data = scale[self._elem_of_entry] * flat
        n = self.n_free
        mat = sp.coo_matrix((data, (self._rows, self._cols)), shape=(n, n)).tocsc()
        return mat

    def stiffness(self, youngs: np.ndarray) -> sp.csc_matrix:
        """``K`` reduced to the free DOFs."""
        return self._assemble(youngs, self._ke_flat)

    def mass(self, densities: np.ndarray) -> sp.csc_matrix:
        """``M`` reduced to the free DOFs."""
        return self._assemble(densities, self._me_flat)

    # -- element-level quadratic forms, used by the adjoint --------------

    def element_quadratic(
        self, left: np.ndarray, right: np.ndarray, which: str = "k"
    ) -> np.ndarray:
        """``(n_elem,)`` array of ``left_e^T A0 right_e`` for every element.

        ``left`` and ``right`` are full-length global vectors and ``which``
        picks ``ke0`` or ``me0``.  This is the per-element term that the
        sensitivity expression needs, computed for all elements at once.
        """
        a0 = self.ke0 if which == "k" else self.me0
        le = left[self.element_dofs]
        re = right[self.element_dofs]
        return np.einsum("ei,ij,ej->e", le, a0, re, optimize=True)
