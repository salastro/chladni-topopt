"""The level-set housekeeping around any design update.

Whatever moves ``Phi`` -- Wang et al.'s Hamilton-Jacobi step or a plain
gradient method on the nodal values -- two things must happen after it: the
clamped corners stay solid, and every few updates ``Phi`` is reset to a signed
distance.  This wrapper does both, so the inner update is any
:class:`~chladni_topopt.optimizers.base.Optimizer` and can be swapped freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..optimizers.base import Optimizer
from .reinit import Reinitializer


@dataclass
class LevelSetUpdate:
    """``inner``'s step, then frozen nodes reset and periodic reinitialisation.

    Parameters
    ----------
    inner:
        The update that actually moves ``Phi``.
    reinit:
        Signed-distance reinitialiser; ``None`` disables it.
    frozen:
        Node indices kept in material, at ``Phi >= frozen_value``.
    reinit_every:
        Updates between reinitialisations.
    """

    inner: Optimizer
    reinit: Reinitializer | None
    frozen: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    frozen_value: float = 0.0
    reinit_every: int = 5
    iteration: int = field(default=0, init=False)

    def update(
        self,
        x: np.ndarray,
        f0: float,
        df0: np.ndarray,
        fi: np.ndarray,
        dfi: np.ndarray,
    ) -> np.ndarray:
        phi = self.inner.update(x, f0, df0, fi, dfi)
        self.iteration += 1
        phi = self.freeze(phi)
        if self.reinit is not None and self.iteration % self.reinit_every == 0:
            phi = self.freeze(self.reinit(phi))
        return phi

    def freeze(self, phi: np.ndarray) -> np.ndarray:
        phi = np.array(phi, dtype=float)
        phi[self.frozen] = np.maximum(phi[self.frozen], self.frozen_value)
        return phi
