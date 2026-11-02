"""Density filter (Eq. 12), threshold projection (Eq. 11) and beta continuation.

The design variables pass through two stages before reaching the material
interpolation::

    x  --filter-->  xtilde  --projection-->  xbar

Both stages are linear-in-shape and elementwise respectively, so the chain rule
of Eq. (18) reduces to ``dC/dx = H^T (dxbar/dxtilde * dC/dxbar)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


@runtime_checkable
class Filter(Protocol):
    """Smooths the raw design field over a neighbourhood."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """``xtilde`` from ``x``."""
        ...

    def backward(self, dc_dxtilde: np.ndarray) -> np.ndarray:
        """Pull a sensitivity back from ``xtilde`` to ``x``."""
        ...


@runtime_checkable
class Projection(Protocol):
    """Elementwise map sharpening the filtered field towards 0/1."""

    def forward(self, xtilde: np.ndarray) -> np.ndarray:
        """``xbar`` from ``xtilde``."""
        ...

    def gradient(self, xtilde: np.ndarray) -> np.ndarray:
        """``dxbar / dxtilde``, elementwise."""
        ...


@dataclass
class DensityFilter:
    """Linear cone filter over a circular neighbourhood, Eq. (12).

    The weight is ``w(x_k) = rmin - ||x_k - x_e||`` for elements inside the
    radius, volume-weighted and row-normalised so the filter preserves a
    uniform field.  Built from element centroids via a KD-tree, so it works on
    any mesh, not only the structured one.
    """

    rmin: float
    centroids: np.ndarray
    volumes: np.ndarray | None = None
    matrix: sp.csr_matrix = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.centroids)
        volumes = np.ones(n) if self.volumes is None else np.asarray(self.volumes)

        tree = cKDTree(self.centroids)
        pairs = tree.query_pairs(self.rmin, output_type="ndarray")
        # Neighbour list including each element's own (zero-distance) entry.
        rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])

        dist = np.linalg.norm(self.centroids[rows] - self.centroids[cols], axis=1)
        weights = (self.rmin - dist) * volumes[cols]

        h = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
        row_sums = np.asarray(h.sum(axis=1)).ravel()
        self.matrix = sp.diags(1.0 / row_sums) @ h

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def backward(self, dc_dxtilde: np.ndarray) -> np.ndarray:
        return self.matrix.T @ dc_dxtilde


@dataclass
class IdentityFilter:
    """No-op filter, useful for isolating the projection in tests."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def backward(self, dc_dxtilde: np.ndarray) -> np.ndarray:
        return dc_dxtilde


@dataclass
class ExponentialThreshold:
    """Smoothed-Heaviside threshold projection, Eq. (11).

    For ``xtilde <= eta``::

        xbar = eta * (exp(-beta * (1 - xtilde/eta)) - (1 - xtilde/eta) * exp(-beta))

    and for ``xtilde > eta``::

        xbar = (1 - eta) * (1 - exp(-beta*(xtilde-eta)/(1-eta))
                            + (xtilde-eta)/(1-eta) * exp(-beta)) + eta

    ``beta`` is mutable because the continuation schedule raises it during the
    run.  At ``beta -> 0`` the map degenerates to the identity.
    """

    eta: float = 0.5
    beta: float = 1.0

    def forward(self, xtilde: np.ndarray) -> np.ndarray:
        eta, beta = self.eta, self.beta
        xt = np.clip(xtilde, 0.0, 1.0)
        lo = xt <= eta

        s_lo = 1.0 - xt / eta
        s_hi = (xt - eta) / (1.0 - eta)
        return np.where(
            lo,
            eta * (np.exp(-beta * s_lo) - s_lo * np.exp(-beta)),
            (1.0 - eta) * (1.0 - np.exp(-beta * s_hi) + s_hi * np.exp(-beta)) + eta,
        )

    def gradient(self, xtilde: np.ndarray) -> np.ndarray:
        eta, beta = self.eta, self.beta
        xt = np.clip(xtilde, 0.0, 1.0)
        lo = xt <= eta

        s_lo = 1.0 - xt / eta
        s_hi = (xt - eta) / (1.0 - eta)
        return np.where(
            lo,
            beta * np.exp(-beta * s_lo) + np.exp(-beta),
            beta * np.exp(-beta * s_hi) + np.exp(-beta),
        )


@runtime_checkable
class Continuation(Protocol):
    """A schedule the driver advances once per iteration.

    The run may only stop once ``at_max`` holds; ``beta`` is recorded in the
    history.
    """

    @property
    def at_max(self) -> bool:
        ...

    @property
    def beta(self) -> float:
        ...

    def step(self, converged: bool) -> bool:
        """Advance by one iteration; return whether the schedule changed."""
        ...


@dataclass
class NoContinuation:
    """No schedule: the design map is fixed, so the run may stop at any time.

    Used by the level-set method, whose smoothed Heaviside has a fixed width.
    """

    at_max: bool = field(default=True, init=False)

    @property
    def beta(self) -> float:
        return float("nan")

    def step(self, converged: bool) -> bool:
        return False


@dataclass
class BetaContinuation:
    """Doubles ``beta`` from ``beta0`` up to ``beta_max``.

    The paper advances the projection parameter "every 50 steps or when the
    optimization process converges", whichever comes first.  ``step`` returns
    ``True`` when ``beta`` was raised, which the driver uses to reset its
    convergence window; the run may only terminate once ``beta`` has reached
    ``beta_max`` and converged there.
    """

    projection: ExponentialThreshold
    beta0: float = 1.0
    beta_max: float = 256.0
    interval: int = 50
    _since_update: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.projection.beta = self.beta0

    @property
    def at_max(self) -> bool:
        return self.projection.beta >= self.beta_max

    @property
    def beta(self) -> float:
        return self.projection.beta

    def step(self, converged: bool) -> bool:
        """Advance the schedule by one iteration; returns whether beta changed."""
        self._since_update += 1
        if self.at_max:
            return False
        if converged or self._since_update >= self.interval:
            self.projection.beta = min(2.0 * self.projection.beta, self.beta_max)
            self._since_update = 0
            return True
        return False
