"""How well a mode's actual nodal line matches the target curve.

The optimization's own indicators do not answer that question.
``Avg(L, phi)`` can be driven down by making the whole neighbourhood of the
target quiet, without a nodal line forming along it, and in this repository it
has reported "converged and feasible" for runs whose pattern looked nothing
like the target.  This module measures the line itself: it extracts the zero
contour of the transverse displacement and compares it with the target curve
in both directions.

* ``traced`` -- fraction of the target lying within ``tol`` of a nodal line.
  Missing hooks, arms that wander off, and ends pulled onto the clamps all
  lower it.
* ``stray`` -- length of nodal line that runs through the target's
  neighbourhood (within ``neighbourhood`` of it) *without* following it,
  relative to the target's length.  An extra line crossing the inside of a U
  raises it; nodal lines elsewhere on the plate are ignored, because the
  formulation does not control them either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import contourpy
import numpy as np
from scipy.spatial import cKDTree

from .bc import DofSet
from .mesh import StructuredQuadMesh


@dataclass(frozen=True)
class NodalFit:
    """Two-way comparison of a nodal pattern with a target curve."""

    #: Fraction of the target within ``tol`` of some nodal line (1 is best).
    traced: float
    #: Mean distance from target points to the nearest nodal line, metres.
    mean_gap: float
    #: 95th percentile of that distance -- the worst stretches, metres.
    worst_gap: float
    #: Nodal-line length near the target but off it, per unit target length.
    stray: float
    tol: float

    def summary(self) -> str:
        return (
            f"traced {100 * self.traced:5.1f}%  stray {self.stray:4.2f}  "
            f"mean gap {1000 * self.mean_gap:5.1f} mm  worst {1000 * self.worst_gap:5.1f} mm"
        )


def nodal_lines(
    mesh: StructuredQuadMesh,
    phi_full: np.ndarray,
    dofs: DofSet | None = None,
    component: int = 0,
    exclude_radius: float | None = None,
    solid: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Zero contours of one nodal component, as ``(N, 2)`` polylines.

    Where the plate is clamped the displacement is identically zero, and the
    contour traces the clamp's outline -- lines that are not nodal lines in
    any useful sense.  With ``dofs`` given, contour points closer than
    ``exclude_radius`` (default 1.5 elements) to a constrained node are
    dropped, splitting the polylines there.

    ``solid`` (element densities, 1 = material) does the same for holes: a
    design with voids still has a displacement field there, but nothing can
    collect on it, so contour points in elements below one half are dropped.
    """
    w = phi_full[mesh.node_dofs(component)].reshape(mesh.nx + 1, mesh.ny + 1).T
    xs = np.linspace(0.0, mesh.lx, mesh.nx + 1)
    ys = np.linspace(0.0, mesh.ly, mesh.ny + 1)
    lines = [
        np.asarray(line)
        for line in contourpy.contour_generator(x=xs, y=ys, z=w).lines(0.0)
        if len(line) > 1
    ]
    has_clamps = dofs is not None and len(dofs.fixed)
    if not has_clamps and solid is None:
        return lines

    if has_clamps:
        radius = 1.5 * max(mesh.dx, mesh.dy) if exclude_radius is None else exclude_radius
        fixed_nodes = np.unique(dofs.fixed // mesh.ndof_per_node)
        tree = cKDTree(mesh.node_coords()[fixed_nodes])

    kept = []
    for line in lines:
        keep = np.ones(len(line), dtype=bool)
        if has_clamps:
            near, _ = tree.query(line)
            keep &= near >= radius
        if solid is not None:
            keep &= _in_material(mesh, solid, line)
        # Split into runs of consecutive kept points.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(int), [0]])))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start > 1:
                kept.append(line[start:stop])
    return kept


def _in_material(mesh: StructuredQuadMesh, solid: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether each point lies in an element with density at least one half."""
    i = np.clip((points[:, 0] / mesh.dx).astype(int), 0, mesh.nx - 1)
    j = np.clip((points[:, 1] / mesh.dy).astype(int), 0, mesh.ny - 1)
    return np.asarray(solid)[i * mesh.ny + j] >= 0.5


def _length(curve: np.ndarray, closed: bool) -> float:
    pts = np.asarray(curve, dtype=float)
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _densify(curves: Sequence[np.ndarray], closed: bool, spacing: float) -> np.ndarray:
    """Points along the curves no further apart than ``spacing``."""
    out = []
    for curve in curves:
        pts = np.asarray(curve, dtype=float)
        if closed and not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[:1]])
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / spacing)), 1)
            t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
        out.append(pts[-1:])
    return np.vstack(out)


def nodal_line_fit(
    mesh: StructuredQuadMesh,
    phi_full: np.ndarray,
    loops: Sequence[np.ndarray] = (),
    open_curves: Sequence[np.ndarray] = (),
    dofs: DofSet | None = None,
    tol: float = 0.06,
    neighbourhood: float = 0.25,
    solid: np.ndarray | None = None,
) -> NodalFit:
    """Compare a mode's nodal lines with a target made of loops and open curves.

    ``tol`` defaults to 0.06 m: the target band plus gap of the paper-resolution
    setup (1 + 2 elements of 0.02 m), i.e. the region the formulation itself
    allows the line to occupy.  ``neighbourhood`` bounds where stray lines
    count.  ``solid`` masks holes out of the nodal lines, as in
    :func:`nodal_lines`.
    """
    spacing = 0.25 * min(mesh.dx, mesh.dy)
    target = np.vstack(
        [p for p in (
            _densify(loops, True, spacing) if len(loops) else None,
            _densify(open_curves, False, spacing) if len(open_curves) else None,
        ) if p is not None]
    )
    lines = nodal_lines(mesh, phi_full, dofs, solid=solid)
    if not lines:
        return NodalFit(traced=0.0, mean_gap=np.inf, worst_gap=np.inf, stray=0.0, tol=tol)

    nodal = np.vstack([_densify([line], False, spacing) for line in lines])
    gap, _ = cKDTree(nodal).query(target)

    # Stray length: nodal segments near the target that do not follow it.
    target_tree = cKDTree(target)
    stray_length = 0.0
    for line in lines:
        mid = 0.5 * (line[1:] + line[:-1])
        length = np.linalg.norm(np.diff(line, axis=0), axis=1)
        d, _ = target_tree.query(mid)
        stray_length += float(length[(d > tol) & (d < neighbourhood)].sum())
    target_length = sum(_length(c, True) for c in loops) + sum(
        _length(c, False) for c in open_curves
    )

    return NodalFit(
        traced=float(np.mean(gap <= tol)),
        mean_gap=float(gap.mean()),
        worst_gap=float(np.percentile(gap, 95)),
        stray=stray_length / target_length,
        tol=tol,
    )
