"""State passed between the analysis, the formulation and the sensitivity step."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .indicators import AVERAGE, VARIANCE
from .selection import SelectionVectors


@dataclass(frozen=True)
class ReferenceIndicators:
    """Indicator values of the *initial* design, ``phi^(0)`` in Eq. (15).

    Every term of the objective and the eigenvector constraints is normalised
    by these, so each one starts at exactly 1 and the weights in a multi-mode
    or multi-objective formulation are comparable.
    """

    avg_shape: float
    avg_surround: float
    var_surround: float

    @classmethod
    def from_mode(
        cls, selection: SelectionVectors, phi: np.ndarray
    ) -> "ReferenceIndicators":
        ref = cls(
            avg_shape=AVERAGE.value(selection.shape, phi),
            avg_surround=AVERAGE.value(selection.surround, phi),
            var_surround=VARIANCE.value(selection.surround, phi),
        )
        for name in ("avg_shape", "avg_surround", "var_surround"):
            if getattr(ref, name) <= 0.0:
                raise ValueError(
                    f"reference indicator {name!r} is not positive; the initial "
                    "mode has no amplitude in that area, so it cannot normalise it"
                )
        return ref


@dataclass(frozen=True)
class TargetMode:
    """One mode being shaped, with its target pattern.

    ``index`` is the position in the spectrum (0-based) that the mode tracker
    currently identifies as this target; ``weight`` is ``w_s`` of Eq. (15).
    """

    index: int
    selection: SelectionVectors
    reference: ReferenceIndicators
    weight: float = 1.0
    value: float = 0.0
    phi_full: np.ndarray = field(default_factory=lambda: np.empty(0))
    phi_reduced: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def frequency(self) -> float:
        """Natural frequency in Hz."""
        return float(np.sqrt(max(self.value, 0.0)) / (2.0 * np.pi))


@dataclass(frozen=True)
class DesignState:
    """The design field at all three stages, plus element volumes."""

    x: np.ndarray
    xtilde: np.ndarray
    xbar: np.ndarray
    volumes: np.ndarray

    @property
    def volume_fraction(self) -> float:
        return float(self.volumes @ self.xbar / self.volumes.sum())


@dataclass(frozen=True)
class Responses:
    """Objective and constraint values with their partial derivatives.

    Row 0 is the objective; rows ``1..m`` are constraints in the form
    ``g_i <= 0``.  The three derivative blocks are *partials*: ``dphi`` and
    ``dlam`` are resolved into design sensitivities by the adjoint, while
    ``dxbar`` is the part that depends on the design directly (the volume
    constraint, and nothing else in these formulations).
    """

    names: tuple[str, ...]
    values: np.ndarray
    dphi: np.ndarray
    dlam: np.ndarray
    dxbar: np.ndarray
    diagnostics: dict[str, float]

    @property
    def objective(self) -> float:
        return float(self.values[0])

    @property
    def constraints(self) -> np.ndarray:
        return self.values[1:]

    @property
    def n_responses(self) -> int:
        return len(self.values)
