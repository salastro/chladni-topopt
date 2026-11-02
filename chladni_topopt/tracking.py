"""Mode tracking via the Modal Assurance Criterion, Eq. (19).

Eigenvalues cross during a dynamic topology optimization: the mode that is
5th in the spectrum at iteration 0 may be 4th or 6th later.  Optimizing "the
5th mode" by index would then silently switch targets mid-run.  The MAC picks
the mode most similar in *shape* to the previous iteration's target instead::

    MAC(phi_r, phi_s) = |phi_r^T phi_s|**2 / ((phi_r^T phi_r)(phi_s^T phi_s))

which is 1 for identical shapes and 0 for orthogonal ones, and is insensitive
to both scaling and sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .eigen import EigenSolution


def mac(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """MAC of one reference vector against each column of ``candidates``."""
    num = (reference @ candidates) ** 2
    den = (reference @ reference) * np.einsum("ik,ik->k", candidates, candidates)
    return num / np.where(den == 0.0, np.inf, den)


@dataclass(frozen=True)
class TrackResult:
    """Which mode was identified as the target, and how confidently."""

    index: int
    mac: float
    switched: bool


@runtime_checkable
class ModeTracker(Protocol):
    """Identifies the target mode within a freshly computed spectrum."""

    index: int

    def select(self, solution: EigenSolution) -> TrackResult:
        ...


@dataclass
class MACTracker:
    """Follows a mode by shape similarity to the previous iteration.

    Parameters
    ----------
    index:
        Position in the spectrum of the target mode at the first call.
    warn_below:
        A selected MAC under this value means no computed mode really
        resembles the previous target -- the spectrum has changed faster than
        the tracker can follow.  Reported via :attr:`last_mac` so the driver
        can surface it rather than failing silently.
    """

    index: int
    warn_below: float = 0.5
    reference: np.ndarray | None = field(default=None, repr=False)
    last_mac: float = 1.0

    def select(self, solution: EigenSolution) -> TrackResult:
        if self.reference is None:
            self.reference = solution.reduced[:, self.index].copy()
            self.last_mac = 1.0
            return TrackResult(index=self.index, mac=1.0, switched=False)

        scores = mac(self.reference, solution.reduced)
        best = int(np.argmax(scores))
        switched = best != self.index

        self.last_mac = float(scores[best])
        self.index = best
        self.reference = solution.reduced[:, best].copy()
        return TrackResult(index=best, mac=self.last_mac, switched=switched)


@dataclass
class FixedIndexTracker:
    """Always returns the same spectral position.  For A/B-testing tracking."""

    index: int
    last_mac: float = 1.0

    def select(self, solution: EigenSolution) -> TrackResult:
        return TrackResult(index=self.index, mac=1.0, switched=False)
