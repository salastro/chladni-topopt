"""Spline offsets -> signed distance of every element from the outline.

:class:`SplineMap` sits in the :class:`~chladni_topopt.filters.Filter` slot of
the analysis chain, so the projection (a smoothed Heaviside), the material
model and the adjoint are the level-set method's, unchanged.  Its ``forward``
turns the design ``u`` into the signed distance of each element centroid from
the spline outline, positive inside; ``backward`` applies the transpose of
that map's Jacobian.

Unlike the linear filters, the map is nonlinear, so ``backward`` needs the
Jacobian at the point ``forward`` last evaluated.  :class:`SplineMap` keeps it
from the last call.  That is safe in this package: ``ChladniAnalysis.evaluate``
always runs ``forward`` and then ``backward`` for the same design.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from matplotlib.path import Path
from scipy.spatial import cKDTree

from .geometry import ClosedBSpline


@dataclass
class SplineMap:
    """``u in [0, 1]^n`` -> element signed distance to the outline.

    Parameters
    ----------
    spline:
        The curve's basis.
    controls, directions:
        Starting control points ``P0`` and their unit move directions ``d``.
    t_lo, t_hi:
        Offset range: ``t = t_lo + u (t_hi - t_lo)``, ``P = P0 + t d``.
    centroids:
        Points the distance is measured at (element centres).
    band:
        Only elements closer than this to the outline get a Jacobian row; the
        smoothed Heaviside is flat further out, so their gradient is zero.
    frozen, frozen_value:
        Elements forced to at least ``frozen_value`` (solid), whatever the
        outline does -- the clamps and the target corridor.
    """

    spline: ClosedBSpline
    controls: np.ndarray
    directions: np.ndarray
    t_lo: float
    t_hi: float
    centroids: np.ndarray
    band: float
    frozen: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    frozen_value: float = 0.0
    _jacobian: sp.csr_matrix | None = field(default=None, init=False, repr=False)

    @property
    def n(self) -> int:
        return len(self.controls)

    def offsets(self, u: np.ndarray) -> np.ndarray:
        return self.t_lo + np.asarray(u) * (self.t_hi - self.t_lo)

    def u_of(self, t: np.ndarray) -> np.ndarray:
        return (np.asarray(t) - self.t_lo) / (self.t_hi - self.t_lo)

    def control_points(self, u: np.ndarray) -> np.ndarray:
        return self.controls + self.offsets(u)[:, None] * self.directions

    def curve(self, u: np.ndarray) -> np.ndarray:
        """``(samples, 2)`` points along the outline."""
        return self.spline.basis @ self.control_points(u)

    def forward(self, u: np.ndarray) -> np.ndarray:
        curve = self.curve(u)
        dist, nearest = cKDTree(curve).query(self.centroids)
        inside = Path(curve).contains_points(self.centroids)
        sd = np.where(inside, dist, -dist)

        # d sd / d c* for the nearest curve sample c*: the unit vector from
        # the point towards the outline when inside, away from it when outside.
        # Both are the outward normal at c*, taken from the actual geometry.
        offset = self.centroids - curve[nearest]
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = offset / dist[:, None]
        unit = np.nan_to_num(unit)
        grad_c = np.where(inside[:, None], -unit, unit)

        held = np.zeros(len(sd), dtype=bool)
        if len(self.frozen):
            held[self.frozen] = sd[self.frozen] < self.frozen_value
            sd[self.frozen] = np.maximum(sd[self.frozen], self.frozen_value)

        rows = np.flatnonzero((np.abs(sd) < self.band) & ~held)
        basis = self.spline.basis[nearest[rows]].tocoo()
        # sd_e moves with control j as B_j(s*) (grad_c . d_j) dt_j/du_j.
        along = np.einsum("ij,ij->i", grad_c[rows][basis.row], self.directions[basis.col])
        values = basis.data * along * (self.t_hi - self.t_lo)
        self._jacobian = sp.csr_matrix(
            (values, (rows[basis.row], basis.col)), shape=(len(sd), self.n)
        )
        return sd

    def backward(self, dc_dsd: np.ndarray) -> np.ndarray:
        if self._jacobian is None:
            raise RuntimeError("SplineMap.backward called before forward")
        return self._jacobian.T @ dc_dsd
