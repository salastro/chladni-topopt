"""A closed, uniform cubic B-spline outline, moved by offsets along fixed directions.

The plate is the region inside one closed curve.  Its control points start on
the plate rectangle and each moves only along its own fixed outward direction,
``P_j = P0_j + t_j d_j``: one scalar per control point, and no reparametrising
as the shape changes.  A uniform cubic B-spline is ``C^2`` everywhere, so the
outline stays smooth whatever the offsets, and each offset moves only the four
curve segments its basis function covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def _cubic_weights(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uniform cubic B-spline weights and their derivatives at local ``u in [0, 1)``."""
    w = np.column_stack([
        (1 - u) ** 3,
        3 * u**3 - 6 * u**2 + 4,
        -3 * u**3 + 3 * u**2 + 3 * u + 1,
        u**3,
    ]) / 6.0
    dw = np.column_stack([
        -3 * (1 - u) ** 2,
        9 * u**2 - 12 * u,
        -9 * u**2 + 6 * u + 3,
        3 * u**2,
    ]) / 6.0
    return w, dw


@dataclass
class ClosedBSpline:
    """Periodic uniform cubic B-spline through ``n_ctrl`` control points.

    ``basis`` maps control points to ``samples`` curve points and ``tangent``
    to the curve's derivatives there, both as sparse ``(samples, n_ctrl)``
    matrices built once: ``curve = basis @ P``.
    """

    n_ctrl: int
    samples: int = 2000
    basis: sp.csr_matrix = field(init=False, repr=False)
    tangent: sp.csr_matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.n_ctrl
        if n < 4:
            raise ValueError("a closed cubic B-spline needs at least 4 control points")
        s = np.arange(self.samples) * n / self.samples  # parameter in [0, n)
        seg = np.floor(s).astype(int)
        w, dw = _cubic_weights(s - seg)
        cols = (seg[:, None] + np.arange(4)[None, :]) % n
        rows = np.repeat(np.arange(self.samples), 4)
        shape = (self.samples, n)
        self.basis = sp.csr_matrix((w.ravel(), (rows, cols.ravel())), shape=shape)
        self.tangent = sp.csr_matrix((dw.ravel(), (rows, cols.ravel())), shape=shape)


def rectangle_controls(
    x0: float, y0: float, lx: float, ly: float, n_ctrl: int
) -> tuple[np.ndarray, np.ndarray]:
    """Control points evenly round a rectangle, counter-clockwise, one on each corner.

    Returns ``(P0, d)``: the points and their unit outward directions (the
    edge normal, or the diagonal at a corner).  Points are shared out between
    the edges in proportion to their length.  A cubic B-spline lies on the
    straight runs between collinear points and rounds each corner off by a
    fraction of the point spacing.
    """
    corners = np.array([[x0, y0], [x0 + lx, y0], [x0 + lx, y0 + ly], [x0, y0 + ly]])
    normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    lengths = np.array([lx, ly, lx, ly])
    counts = np.maximum(1, np.round(n_ctrl * lengths / lengths.sum()).astype(int))
    counts[np.argmax(counts)] += n_ctrl - counts.sum()

    points, dirs = [], []
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        for i in range(counts[k]):
            points.append(a + (b - a) * i / counts[k])
            if i == 0:  # the corner itself
                diag = normals[k] + normals[k - 1]
                dirs.append(diag / np.linalg.norm(diag))
            else:
                dirs.append(normals[k])
    return np.array(points), np.array(dirs)
