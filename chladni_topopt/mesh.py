"""Structured quadrilateral mesh for a rectangular plate.

Node and element numbering are column-major in ``x``: node ``(i, j)`` with
``i`` along ``x`` and ``j`` along ``y`` has index ``i * (ny + 1) + j``.  The
same convention is used for elements, which keeps the reshape to an
``(nx, ny)`` image trivial in :mod:`chladni_topopt.postprocess`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredQuadMesh:
    """A uniform grid of ``nx * ny`` rectangular elements over ``[0, lx] x [0, ly]``.

    Parameters
    ----------
    lx, ly:
        Plate dimensions in metres.
    nx, ny:
        Number of elements along each axis.
    ndof_per_node:
        Degrees of freedom carried by each node.  Three for the
        Reissner-Mindlin plate ``(w, theta_x, theta_y)``.
    """

    lx: float
    ly: float
    nx: int
    ny: int
    ndof_per_node: int = 3

    # -- counts ---------------------------------------------------------

    @property
    def n_elem(self) -> int:
        return self.nx * self.ny

    @property
    def n_node(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_dof(self) -> int:
        return self.n_node * self.ndof_per_node

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    @property
    def elem_area(self) -> float:
        return self.dx * self.dy

    # -- geometry -------------------------------------------------------

    def node_coords(self) -> np.ndarray:
        """``(n_node, 2)`` array of node positions."""
        xs = np.linspace(0.0, self.lx, self.nx + 1)
        ys = np.linspace(0.0, self.ly, self.ny + 1)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def elem_centroids(self) -> np.ndarray:
        """``(n_elem, 2)`` array of element centre positions."""
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def node_index(self, i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray | int:
        """Node number from grid indices ``i`` (along x) and ``j`` (along y)."""
        return i * (self.ny + 1) + j

    # -- connectivity ---------------------------------------------------

    def connectivity(self) -> np.ndarray:
        """``(n_elem, 4)`` node numbers, counter-clockwise from the lower left.

        The ordering matches the reference element used in
        :mod:`chladni_topopt.elements.mindlin_q4`: ``(-1,-1), (1,-1), (1,1), (-1,1)``.
        """
        ei, ej = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        ei, ej = ei.ravel(), ej.ravel()
        n0 = self.node_index(ei, ej)
        n1 = self.node_index(ei + 1, ej)
        n2 = self.node_index(ei + 1, ej + 1)
        n3 = self.node_index(ei, ej + 1)
        return np.column_stack([n0, n1, n2, n3])

    def element_dofs(self) -> np.ndarray:
        """``(n_elem, 4 * ndof_per_node)`` global DOF numbers per element."""
        conn = self.connectivity()
        nd = self.ndof_per_node
        return (conn[:, :, None] * nd + np.arange(nd)[None, None, :]).reshape(
            self.n_elem, 4 * nd
        )

    def node_dofs(self, component: int = 0) -> np.ndarray:
        """Global DOF numbers of one nodal component for every node.

        ``component=0`` selects the transverse displacement ``w``, which is
        what the Chladni selection vectors act on.
        """
        return np.arange(self.n_node) * self.ndof_per_node + component
