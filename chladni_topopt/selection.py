"""Selection vectors ``L`` and ``L~`` from a target pattern (Fig. 2b).

A target Chladni pattern is given as plain arrays of 2-D points in plate
coordinates: closed ``loops`` and/or ``open_curves``.  A nodal line either
closes on itself or ends on the plate boundary -- it cannot stop in mid-plate --
so an open curve should run all the way to the edges.  Nodes are sorted by
their distance to the target into three zones:

* ``|d| <= band``                              -> ``L``, the "shape area"
* ``band < |d| <= band + gap``                 -> unmarked
* ``band + gap < |d| <= band + gap + surround`` -> ``L~``, the "surrounding area"

The gap is what lets the optimizer place the nodal line near, rather than
exactly on, the target curve, and the surrounding zone deliberately covers
*both* sides of the loop: the objective drives it to a large, uniform
amplitude, which forces the mode to change sign across the target and so puts
a nodal line there.

Only the transverse DOF of each node is selected, and constrained nodes are
excluded -- their displacement is identically zero, so including them would
bias both indicators for reasons that have nothing to do with the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.path import Path

from .bc import DofSet
from .mesh import StructuredQuadMesh


@dataclass(frozen=True)
class SelectionVectors:
    """The pair ``(L, L~)`` plus the geometry they came from."""

    shape: np.ndarray
    surround: np.ndarray
    #: Distance to the target, signed so the two sides of the nodal line have
    #: opposite signs: negative inside a closed loop, and for an open curve
    #: negative on its left (walking along the points in order).
    signed_distance: np.ndarray
    loops: tuple[np.ndarray, ...]
    open_curves: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if not self.shape.any():
            raise ValueError("selection vector L is empty; widen `band`")
        if not self.surround.any():
            raise ValueError("selection vector L~ is empty; widen `surround`")

    @property
    def n_shape(self) -> int:
        return int(self.shape.sum())

    @property
    def n_surround(self) -> int:
        return int(self.surround.sum())


def _segments(points: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Start and end points of a polyline's segments.

    A closed loop gets its closing segment added unless the last point already
    repeats the first; an open curve never does.
    """
    points = np.asarray(points, dtype=float)
    if closed and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[:1]])
    return points[:-1], points[1:]


def _nearest_on_segments(
    points: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to its nearest segment, and that segment's index."""
    seg = b - a
    length2 = np.einsum("ij,ij->i", seg, seg)
    length2 = np.where(length2 == 0.0, 1.0, length2)

    # Projection parameter of every point onto every segment.
    diff = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", diff, seg) / length2, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    index = dist.argmin(axis=1)
    return dist[np.arange(len(points)), index], index


def _distance_to_loops(points: np.ndarray, loops: Sequence[np.ndarray]) -> np.ndarray:
    """Minimum euclidean distance from each point to any loop's boundary."""
    best = np.full(len(points), np.inf)
    for loop in loops:
        dist, _ = _nearest_on_segments(points, *_segments(loop, closed=True))
        best = np.minimum(best, dist)
    return best


def _inside_loops(points: np.ndarray, loops: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean mask of points enclosed by an odd number of loops."""
    inside = np.zeros(len(points), dtype=bool)
    for loop in loops:
        inside ^= Path(loop).contains_points(points)
    return inside


def _signed_distance_to_open_curves(
    points: np.ndarray, curves: Sequence[np.ndarray]
) -> np.ndarray:
    """Distance to the nearest open curve, negative on its left-hand side.

    "Left" is taken from the nearest segment, walking along the points in
    order.  For a curve that runs edge to edge this splits the plate into its
    two sides consistently, which is all the sign is used for.
    """
    best = np.full(len(points), np.inf)
    signed = np.full(len(points), np.inf)
    for curve in curves:
        a, b = _segments(curve, closed=False)
        dist, index = _nearest_on_segments(points, a, b)
        seg = b[index] - a[index]
        rel = points - a[index]
        cross = seg[:, 0] * rel[:, 1] - seg[:, 1] * rel[:, 0]
        closer = dist < best
        best = np.where(closer, dist, best)
        signed = np.where(closer, np.where(cross > 0.0, -dist, dist), signed)
    return signed


def signed_distance_to_target(
    points: np.ndarray,
    loops: Sequence[np.ndarray] = (),
    open_curves: Sequence[np.ndarray] = (),
) -> np.ndarray:
    """Distance to the nearest part of the target, with a side-of-line sign.

    Closed loops sign by inside (negative) and outside; open curves by left
    (negative) and right.  Each point takes the sign of whichever part of the
    target it is nearest to.
    """
    if not len(loops) and not len(open_curves):
        raise ValueError("a target needs at least one loop or open curve")
    result = np.full(len(points), np.inf)
    if len(loops):
        dist = _distance_to_loops(points, loops)
        result = np.where(_inside_loops(points, loops), -dist, dist)
    if len(open_curves):
        curve = _signed_distance_to_open_curves(points, open_curves)
        result = np.where(np.abs(curve) < np.abs(result), curve, result)
    return result


def signed_distance_to_loops(
    points: np.ndarray, loops: Sequence[np.ndarray]
) -> np.ndarray:
    """Distance to closed loops, negative inside."""
    return signed_distance_to_target(points, loops=loops)


def build_selection_vectors(
    mesh: StructuredQuadMesh,
    dofs: DofSet,
    loops: Sequence[np.ndarray],
    band: float,
    gap: float,
    surround: float,
    transverse_dof: int = 0,
    open_curves: Sequence[np.ndarray] = (),
) -> SelectionVectors:
    """Assemble ``L`` and ``L~`` as full-length 0/1 DOF vectors.

    Parameters
    ----------
    loops:
        Closed target curves.  May be empty when ``open_curves`` is given.
    band, gap, surround:
        Zone half-widths in metres, measured outwards from the target curve.
        See :func:`widths_in_elements` to specify them in element counts.
    transverse_dof:
        Index of ``w`` within a node's DOF block.
    open_curves:
        Target curves that do not close.  A nodal line cannot end in
        mid-plate, so each should run to the plate boundary; nothing enforces
        that, because a curve stopping short is still a legitimate (if
        unreachable) request.
    """
    coords = mesh.node_coords()
    signed = signed_distance_to_target(coords, loops, open_curves)
    dist = np.abs(signed)

    in_shape = dist <= band
    in_surround = (dist > band + gap) & (dist <= band + gap + surround)

    node_dofs = mesh.node_dofs(transverse_dof)
    free_mask = np.zeros(mesh.n_dof, dtype=bool)
    free_mask[dofs.free] = True
    usable = free_mask[node_dofs]

    shape_vec = np.zeros(mesh.n_dof)
    surround_vec = np.zeros(mesh.n_dof)
    shape_vec[node_dofs[in_shape & usable]] = 1.0
    surround_vec[node_dofs[in_surround & usable]] = 1.0

    return SelectionVectors(
        shape=shape_vec,
        surround=surround_vec,
        signed_distance=signed,
        loops=tuple(np.asarray(loop, dtype=float) for loop in loops),
        open_curves=tuple(np.asarray(c, dtype=float) for c in open_curves),
    )


def linear_profile_shape_ratio(
    mesh: StructuredQuadMesh,
    selection: SelectionVectors,
    phi_reference: np.ndarray,
) -> float:
    """Scale estimate for how small ``Avg(L, phi) / Avg(L, phi0)`` can get.

    ``L`` is a band of finite width, so a mode whose nodal line runs exactly
    along the target still carries amplitude elsewhere in the band.  Modelling
    the mode as linear through the nodal line, ``phi ~ A * d``, and fixing
    ``A`` by requiring the surrounding amplitude to match the reference design
    gives

        Avg(L, phi) / Avg(L, phi0)  ~  Avg(L~, phi0) <d^2>_L
                                       ---------------------
                                        Avg(L, phi0)  <d^2>_L~

    which scales as the square of the band width.  That quadratic scaling is
    the useful part: doubling ``band`` raises the achievable ratio by roughly
    four, so the band width, not the optimizer, is usually what decides whether
    a given ``eps`` is within reach.

    This is an **estimate, not a bound**, on two counts.  It assumes the mode
    is linear across both zones, whereas a real mode is free to stay flat
    inside the band and turn up sharply outside it, which beats the estimate.
    And it assumes the surrounding amplitude holds at the reference level
    (objective ``C <= 1``); a design that lets ``Avg(L~, phi)`` fall scales the
    whole profile down and can go below the estimate even after correcting by
    ``C``.  Both have been observed in practice.

    Use it to sanity-check ``eps`` before committing to a long run -- if the
    estimate sits well above ``eps``, expect a hard time -- not to declare a
    configuration infeasible.
    """
    from .indicators import AVERAGE

    node_dofs = mesh.node_dofs()
    dist = np.abs(selection.signed_distance)
    in_shape = selection.shape[node_dofs] > 0
    in_surround = selection.surround[node_dofs] > 0

    mean_d2_shape = float((dist[in_shape] ** 2).mean())
    mean_d2_surround = float((dist[in_surround] ** 2).mean())

    avg_shape = AVERAGE.value(selection.shape, phi_reference)
    avg_surround = AVERAGE.value(selection.surround, phi_reference)
    return (avg_surround * mean_d2_shape) / (avg_shape * mean_d2_surround)


def widths_in_elements(
    mesh: StructuredQuadMesh, band: float, gap: float, surround: float
) -> tuple[float, float, float]:
    """Convert zone widths given in element counts to metres."""
    unit = 0.5 * (mesh.dx + mesh.dy)
    return band * unit, gap * unit, surround * unit
