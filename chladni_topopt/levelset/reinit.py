"""Reinitialisation: reset ``Phi`` to the signed distance from its zero contour.

The upwind update moves the zero level set correctly but lets ``|grad Phi|``
drift away from 1, and the smoothed Heaviside of Eq. (40) only has its
intended width while ``Phi`` is a distance (Eq. 43).  Wang et al. take this
from the level-set literature without giving a scheme; here the zero contour
is extracted explicitly and the distance to it measured directly, which needs
nothing beyond the contouring already used for the nodal-line metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import contourpy
import numpy as np
from scipy.spatial import cKDTree

from ..mesh import StructuredQuadMesh
from ..metrics import _densify


@runtime_checkable
class Reinitializer(Protocol):
    """Restores ``Phi`` to a signed-distance function without moving its zero set."""

    def __call__(self, phi: np.ndarray) -> np.ndarray:
        ...


@dataclass
class ContourReinitializer:
    """Distance to the densified ``Phi = 0`` contour, signed by the old ``Phi``.

    The contour is the bilinear one through the nodal values, so nodes keep
    their side and the boundary moves by at most the densification spacing.
    """

    mesh: StructuredQuadMesh
    #: Contour point spacing, as a fraction of the smaller element side.
    spacing: float = 0.25

    def __call__(self, phi: np.ndarray) -> np.ndarray:
        mesh = self.mesh
        z = phi.reshape(mesh.nx + 1, mesh.ny + 1).T
        xs = np.linspace(0.0, mesh.lx, mesh.nx + 1)
        ys = np.linspace(0.0, mesh.ly, mesh.ny + 1)
        lines = [
            np.asarray(line)
            for line in contourpy.contour_generator(x=xs, y=ys, z=z).lines(0.0)
            if len(line) > 1
        ]
        if not lines:  # no boundary inside the plate: nothing to measure from
            return phi.copy()
        points = _densify(lines, False, self.spacing * min(mesh.dx, mesh.dy))
        dist, _ = cKDTree(points).query(mesh.node_coords())
        return np.where(phi >= 0.0, dist, -dist)
