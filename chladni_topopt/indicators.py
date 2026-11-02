"""Amplitude indicators over a selected area, Eqs. (13)-(14).

Both act on the Hadamard square ``phi o phi``, so they are blind to the sign of
the eigenvector -- which is what makes them usable as design criteria, since
the sign of a mode carries phase, not pattern.

The average drives the *formation* of a nodal line (small on the target curve,
large around it); the variance makes the pattern *sharp* by forcing the
surrounding amplitude to be uniform rather than merely large somewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Indicator(Protocol):
    """A scalar measure of a mode's amplitude over a selected area."""

    name: str

    def value(self, selector: np.ndarray, phi: np.ndarray) -> float:
        ...

    def gradient(self, selector: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """``d value / d phi``, full length."""
        ...


class Average:
    """``Avg(L, phi) = L^T (phi o phi) / sum(L)``, Eq. (13)."""

    name = "avg"

    def value(self, selector: np.ndarray, phi: np.ndarray) -> float:
        return float(selector @ (phi * phi) / selector.sum())

    def gradient(self, selector: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return 2.0 * selector * phi / selector.sum()


class Variance:
    """``Var(L, phi) = L^T ((phi o phi) - mu)**2 / sum(L)``, Eq. (14).

    ``mu`` is the average over the same area.  Its dependence on ``phi`` drops
    out of the gradient: the chain-rule term carries a factor
    ``sum_i L_i (phi_i**2 - mu)``, which is zero by the definition of ``mu``.
    """

    name = "var"

    def value(self, selector: np.ndarray, phi: np.ndarray) -> float:
        total = selector.sum()
        sq = phi * phi
        mu = selector @ sq / total
        return float(selector @ (sq - mu) ** 2 / total)

    def gradient(self, selector: np.ndarray, phi: np.ndarray) -> np.ndarray:
        total = selector.sum()
        sq = phi * phi
        mu = selector @ sq / total
        return 4.0 * selector * phi * (sq - mu) / total


AVERAGE = Average()
VARIANCE = Variance()
