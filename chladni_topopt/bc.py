"""Boundary conditions: which global degrees of freedom are prescribed to zero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .mesh import StructuredQuadMesh


@runtime_checkable
class BoundaryCondition(Protocol):
    """Selects the constrained degrees of freedom of a mesh."""

    def fixed_dofs(self, mesh: StructuredQuadMesh) -> np.ndarray:
        """Sorted array of global DOF indices held at zero."""
        ...


@dataclass(frozen=True)
class CornerClamps:
    """Fully fixed square patches at the four corners of a rectangular plate.

    Section 4 of the paper: "The shape of the fully fixed area at the corners
    is a square whose side length is 1/10 of the short side of the rectangular
    plate."  Every DOF of every node inside those squares is constrained.
    """

    fraction: float = 0.1

    def fixed_dofs(self, mesh: StructuredQuadMesh) -> np.ndarray:
        side = self.fraction * min(mesh.lx, mesh.ly)
        coords = mesh.node_coords()
        x, y = coords[:, 0], coords[:, 1]
        # Half an element of slack so nodes exactly on the patch edge count.
        tol = 1e-9 * side
        near_x0 = x <= side + tol
        near_x1 = x >= mesh.lx - side - tol
        near_y0 = y <= side + tol
        near_y1 = y >= mesh.ly - side - tol
        in_corner = (near_x0 | near_x1) & (near_y0 | near_y1)

        nodes = np.flatnonzero(in_corner)
        nd = mesh.ndof_per_node
        return np.sort((nodes[:, None] * nd + np.arange(nd)[None, :]).ravel())


@dataclass(frozen=True)
class ClampedEdges:
    """All four edges fully clamped.  Provided as a drop-in alternative."""

    def fixed_dofs(self, mesh: StructuredQuadMesh) -> np.ndarray:
        coords = mesh.node_coords()
        x, y = coords[:, 0], coords[:, 1]
        tol = 1e-9 * min(mesh.lx, mesh.ly)
        on_edge = (
            (x <= tol) | (x >= mesh.lx - tol) | (y <= tol) | (y >= mesh.ly - tol)
        )
        nodes = np.flatnonzero(on_edge)
        nd = mesh.ndof_per_node
        return np.sort((nodes[:, None] * nd + np.arange(nd)[None, :]).ravel())


@dataclass(frozen=True)
class SimplySupportedEdges:
    """Transverse displacement fixed on all four edges; rotations free.

    Used by the element verification test, where the Mindlin plate has a
    closed-form solution.
    """

    def fixed_dofs(self, mesh: StructuredQuadMesh) -> np.ndarray:
        coords = mesh.node_coords()
        x, y = coords[:, 0], coords[:, 1]
        tol = 1e-9 * min(mesh.lx, mesh.ly)
        on_edge = (
            (x <= tol) | (x >= mesh.lx - tol) | (y <= tol) | (y >= mesh.ly - tol)
        )
        return np.sort(np.flatnonzero(on_edge) * mesh.ndof_per_node)


@dataclass(frozen=True)
class DofSet:
    """Bookkeeping for the free/fixed split.

    Eigenvectors are solved on the free DOFs but stored full-length with zeros
    on the fixed ones, so the Chladni selection vectors can index the global
    DOF numbering directly.
    """

    n_dof: int
    fixed: np.ndarray

    @classmethod
    def from_mesh(
        cls, mesh: StructuredQuadMesh, bc: BoundaryCondition
    ) -> "DofSet":
        return cls(n_dof=mesh.n_dof, fixed=bc.fixed_dofs(mesh))

    @property
    def free(self) -> np.ndarray:
        mask = np.ones(self.n_dof, dtype=bool)
        mask[self.fixed] = False
        return np.flatnonzero(mask)

    @property
    def n_free(self) -> int:
        return self.n_dof - len(self.fixed)

    def reduced_index(self) -> np.ndarray:
        """``(n_dof,)`` map to reduced numbering; ``-1`` for fixed DOFs."""
        idx = np.full(self.n_dof, -1, dtype=np.int64)
        idx[self.free] = np.arange(self.n_free)
        return idx

    def expand(self, reduced: np.ndarray) -> np.ndarray:
        """Scatter a reduced vector (or set of column vectors) to full length."""
        if reduced.ndim == 1:
            full = np.zeros(self.n_dof)
            full[self.free] = reduced
            return full
        full = np.zeros((self.n_dof, reduced.shape[1]))
        full[self.free] = reduced
        return full
