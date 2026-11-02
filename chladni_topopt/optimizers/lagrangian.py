"""Powell-Hestenes-Rockafellar augmented Lagrangian for inequality constraints.

Shared by every first-order update that cannot handle constraints itself:
AdamW on densities, and the Hamilton-Jacobi level-set step, which needs one
combined sensitivity to turn into a boundary speed.  For

    min f0(x)  s.t.  fi(x) <= 0

the descent direction is the gradient of

    L(x) = f0(x) + sum_i (max(0, lam_i + rho * f_i(x))**2 - lam_i**2) / (2 rho)

that is ``df0 + sum_i max(0, lam_i + rho f_i) * dfi``, and every
``dual_interval`` steps the multipliers take the dual ascent step
``lam_i <- max(0, lam_i + rho f_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AugmentedLagrangian:
    """Multipliers and the combined gradient for ``m`` inequality constraints."""

    m: int
    rho: float = 10.0
    dual_interval: int = 5
    multipliers: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.multipliers = np.zeros(self.m)

    def gradient(self, df0: np.ndarray, fi: np.ndarray, dfi: np.ndarray) -> np.ndarray:
        """Gradient of the augmented Lagrangian at the current multipliers."""
        weights = np.maximum(0.0, self.multipliers + self.rho * fi)
        return df0 + weights @ dfi

    def dual_step(self, fi: np.ndarray, iteration: int) -> None:
        """Dual ascent, on every ``dual_interval``-th primal iteration (1-based).

        ``fi`` must be the constraint values at the point the gradient was
        taken, i.e. the one the driver just analysed.
        """
        if self.m and iteration % self.dual_interval == 0:
            self.multipliers = np.maximum(0.0, self.multipliers + self.rho * fi)
