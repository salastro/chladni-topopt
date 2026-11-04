"""A plate inside a larger design domain, so its outline can grow as well as shrink.

On a mesh that is exactly the plate, a boundary-geometry design can only move
the outline inward.  Embedding the plate in a domain ``margin`` wider on every
side, with void around it, lets the outline move both ways.  The supports do
not move with the mesh: the clamped squares stay at the original plate's
corners (:class:`InsetCornerClamps`), so whatever shape the plate takes, it is
held where the physical plate is held.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MeshConfig
from .mesh import StructuredQuadMesh


@dataclass(frozen=True)
class InsetCornerClamps:
    """Fully fixed squares at the corners of a rectangle inside the mesh.

    :class:`~chladni_topopt.bc.CornerClamps` with the rectangle
    ``[x0, x1] x [y0, y1]`` in place of the mesh's own extent.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    side: float

    def fixed_dofs(self, mesh: StructuredQuadMesh) -> np.ndarray:
        x, y = mesh.node_coords().T
        tol = 1e-9 * self.side
        in_x = (np.abs(x - self.x0 - 0.5 * self.side) <= 0.5 * self.side + tol) | (
            np.abs(x - self.x1 + 0.5 * self.side) <= 0.5 * self.side + tol
        )
        in_y = (np.abs(y - self.y0 - 0.5 * self.side) <= 0.5 * self.side + tol) | (
            np.abs(y - self.y1 + 0.5 * self.side) <= 0.5 * self.side + tol
        )
        nodes = np.flatnonzero(in_x & in_y)
        nd = mesh.ndof_per_node
        return np.sort((nodes[:, None] * nd + np.arange(nd)[None, :]).ravel())


@dataclass(frozen=True)
class PlateInDomain:
    """The paper's plate, ``margin`` metres inside a void-filled design domain.

    ``margin = 0`` is the plate on its own, as in the density method.
    Coordinates of targets and initial shapes are given in plate coordinates
    and moved into the domain with :meth:`shift`.
    """

    plate: tuple[float, float] = (2.0, 2.4)
    margin: float = 0.3
    element: float = 0.02
    #: Clamped square side as a fraction of the plate's short side (the paper's 1/10).
    clamp_fraction: float = 0.1

    @property
    def size(self) -> tuple[float, float]:
        return self.plate[0] + 2 * self.margin, self.plate[1] + 2 * self.margin

    @property
    def origin(self) -> np.ndarray:
        """Where the plate's lower-left corner sits in the domain."""
        return np.array([self.margin, self.margin])

    def mesh(self, thickness: float = 0.2) -> MeshConfig:
        lx, ly = self.size
        return MeshConfig(
            lx=lx, ly=ly, nx=round(lx / self.element), ny=round(ly / self.element),
            thickness=thickness,
        )

    def clamps(self) -> InsetCornerClamps:
        (x0, y0), (px, py) = self.origin, self.plate
        return InsetCornerClamps(
            x0=x0, y0=y0, x1=x0 + px, y1=y0 + py, side=self.clamp_fraction * min(px, py)
        )

    def shift(self, curves: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Curves in plate coordinates, moved into domain coordinates."""
        return [np.asarray(c, dtype=float) + self.origin for c in curves]

    def plate_sdf(self, points: np.ndarray) -> np.ndarray:
        """Signed distance to the plate rectangle, positive inside."""
        half = 0.5 * np.asarray(self.plate)
        q = np.abs(np.asarray(points) - self.origin - half) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return -(outside + inside)
