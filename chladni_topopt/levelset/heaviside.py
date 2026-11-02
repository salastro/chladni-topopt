"""Wang et al.'s smoothed Heaviside and Dirac functions, Eqs. (39)-(40).

On a fixed finite-element grid the boundary ``Phi = 0`` cuts through elements,
so the step ``H(Phi)`` is replaced by a cubic ramp of half-width ``width``::

    H(p) = alpha                                                     p < -width
         = 3(1 - alpha)/4 (p/width - p^3 / (3 width^3)) + (1 + alpha)/2   |p| <= width
         = 1                                                         p > width

and ``delta = dH/dp`` is the matching parabola.  As a
:class:`~chladni_topopt.filters.Projection` it maps the element value of
``Phi`` to a physical density, so the existing analysis and adjoint take it
unchanged; its gradient is the ``delta(Phi)`` that confines every sensitivity
to a band around the boundary, which is Wang's narrow band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WangHeaviside:
    """Eq. (40) as a projection, Eq. (39) as its gradient.

    ``alpha`` is Wang's floor that keeps the void stiffness non-zero.  It is 0
    by default because the floor belongs to the material model here (see
    :class:`~chladni_topopt.material.SolidVoidErsatz`); a non-zero value would
    apply it twice.
    """

    width: float
    alpha: float = 0.0

    def forward(self, phi: np.ndarray) -> np.ndarray:
        a, w = self.alpha, self.width
        p = np.clip(phi, -w, w) / w
        return 0.75 * (1.0 - a) * (p - p**3 / 3.0) + 0.5 * (1.0 + a)

    def gradient(self, phi: np.ndarray) -> np.ndarray:
        a, w = self.alpha, self.width
        p = phi / w
        return np.where(np.abs(p) <= 1.0, 0.75 * (1.0 - a) / w * (1.0 - p**2), 0.0)
