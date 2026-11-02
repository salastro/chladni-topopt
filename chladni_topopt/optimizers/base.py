"""Optimizer interface.

An optimizer consumes the current point and the objective/constraint values and
gradients, and returns the next point.  It owns whatever state it needs across
iterations (MMA keeps two previous designs and its asymptotes), so the driver
stays a plain loop.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Optimizer(Protocol):
    """Gradient-based update for a bound-constrained, inequality-constrained problem.

    The problem is always stated as::

        min f0(x)  s.t.  fi(x) <= 0,  xmin <= x <= xmax
    """

    def update(
        self,
        x: np.ndarray,
        f0: float,
        df0: np.ndarray,
        fi: np.ndarray,
        dfi: np.ndarray,
    ) -> np.ndarray:
        """Return the next design point.

        Parameters
        ----------
        x:
            Current design, ``(n,)``.
        f0, df0:
            Objective value and its gradient ``(n,)``.
        fi, dfi:
            Constraint values ``(m,)`` and their gradients ``(m, n)``.
        """
        ...
