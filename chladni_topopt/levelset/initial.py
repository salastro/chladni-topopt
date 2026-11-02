"""Initial level-set functions, and the nodes the boundary may not reach.

A level-set boundary only moves; Wang et al.'s method has no mechanism for
opening a hole where there is none.  Every topology the run can reach must
therefore be reachable by moving and merging the holes it starts with, which
is why Section 8 starts from perforated plates -- and why those converge
fastest.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..bc import DofSet
from ..mesh import StructuredQuadMesh
from ..selection import SelectionVectors


def perforated(
    mesh: StructuredQuadMesh,
    holes: tuple[int, int] = (6, 7),
    radius: float = 0.35,
) -> np.ndarray:
    """Signed distance to a regular array of circular holes; positive in material.

    ``holes`` is the number of holes along x and y, centred in equal cells.
    ``radius`` is a fraction of the smaller cell side, so the starting solid
    fraction is ``1 - pi radius^2 * (smaller/larger side)`` -- about 0.6 at the
    default on a square cell, and independent of the mesh.
    """
    nhx, nhy = holes
    cell_x, cell_y = mesh.lx / nhx, mesh.ly / nhy
    r = radius * min(cell_x, cell_y)
    cx, cy = np.meshgrid(
        (np.arange(nhx) + 0.5) * cell_x, (np.arange(nhy) + 0.5) * cell_y, indexing="ij"
    )
    centres = np.column_stack([cx.ravel(), cy.ravel()])
    dist, _ = cKDTree(centres).query(mesh.node_coords())
    return dist - r


def live_edge(mesh: StructuredQuadMesh, phi: np.ndarray, margin: float) -> np.ndarray:
    """``phi`` with the plate's outline made a moving boundary too.

    A level set only moves where ``Phi = 0``.  With ``Phi > 0`` all along the
    plate edge the outline is fixed: the reference domain's edge, with
    zero-flux conditions, not a boundary of the design.  Taking the smaller of
    ``phi`` and the distance to the edge minus ``margin`` puts a zero contour
    ``margin`` inside the edge all the way round, so the outline gets a speed
    like any hole boundary and can be pushed in, notched or straightened.  It
    cannot grow past the reference domain.
    """
    xy = mesh.node_coords()
    to_edge = np.minimum.reduce(
        [xy[:, 0], mesh.lx - xy[:, 0], xy[:, 1], mesh.ly - xy[:, 1]]
    )
    return np.minimum(phi, to_edge - margin)


def frozen_nodes(
    mesh: StructuredQuadMesh, dofs: DofSet, margin: float = 0.0
) -> np.ndarray:
    """Nodes that must stay solid: the clamped ones, plus ``margin`` around them.

    A hole over a clamp would leave the support holding nothing, and the
    eigenproblem would no longer describe the clamped plate.
    """
    fixed = np.unique(dofs.fixed // mesh.ndof_per_node)
    if not len(fixed):
        return fixed
    coords = mesh.node_coords()
    if margin <= 0.0:
        return fixed
    dist, _ = cKDTree(coords[fixed]).query(coords)
    return np.flatnonzero(dist <= margin)


def target_corridor(
    mesh: StructuredQuadMesh, selection: SelectionVectors, half_width: float
) -> np.ndarray:
    """Nodes within ``half_width`` of a target curve, to be kept solid.

    A nodal line is only visible where there is plate under it: a hole across
    the target breaks the pattern exactly where it should show.  Freezing a
    solid corridor along the target, like the clamps, keeps every hole off
    it.  The distance is the selection's own signed distance, so loops and
    open curves are handled alike.
    """
    dist = selection.signed_distance  # one value per node
    return np.flatnonzero(np.abs(dist) <= half_width)
