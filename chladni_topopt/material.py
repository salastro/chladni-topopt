"""Material interpolation: physical density -> element Young's modulus and density."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MaterialInterpolation(Protocol):
    """Maps physical densities ``xbar`` to element material properties."""

    def youngs(self, xbar: np.ndarray) -> np.ndarray:
        """Element Young's moduli."""
        ...

    def density(self, xbar: np.ndarray) -> np.ndarray:
        """Element mass densities."""
        ...

    def d_youngs(self, xbar: np.ndarray) -> np.ndarray:
        """``dE / dxbar``, elementwise."""
        ...

    def d_density(self, xbar: np.ndarray) -> np.ndarray:
        """``drho / dxbar``, elementwise."""
        ...


@dataclass(frozen=True)
class SIMPTwoMaterial:
    """Two-material SIMP interpolation, Eq. (10).

    ``xbar = 1`` is the strong material (aluminium), ``xbar = 0`` the soft one
    (epoxy).  Young's modulus is penalised, mass is not -- penalising only the
    stiffness is what pushes the design towards 0/1 in an eigenvalue problem.

    Note that with two real materials the modulus ratio is a modest ``E1 / E2``
    (about 68 for aluminium over epoxy), not the ``1e-9`` of a solid-void
    problem.  Low-density regions therefore carry real stiffness and the usual
    spurious localised eigenmodes do not appear.
    """

    e1: float
    e2: float
    rho1: float
    rho2: float
    penal: float = 3.0

    def youngs(self, xbar: np.ndarray) -> np.ndarray:
        return xbar**self.penal * self.e1 + (1.0 - xbar**self.penal) * self.e2

    def density(self, xbar: np.ndarray) -> np.ndarray:
        return xbar * self.rho1 + (1.0 - xbar) * self.rho2

    def d_youngs(self, xbar: np.ndarray) -> np.ndarray:
        return self.penal * xbar ** (self.penal - 1.0) * (self.e1 - self.e2)

    def d_density(self, xbar: np.ndarray) -> np.ndarray:
        return np.full_like(xbar, self.rho1 - self.rho2)


@dataclass(frozen=True)
class SolidVoidErsatz:
    """One material and void, the void modelled as a very soft, very light solid.

    ``xbar = 1`` is the plate material and ``xbar = 0`` a hole.  This is the
    "ersatz material" of fixed-grid level-set methods: a hole cannot be removed
    from the mesh, so it keeps a small fraction of the stiffness and mass::

        E   = e   * (e_void   + (1 - e_void)   * xbar)
        rho = rho * (rho_void + (1 - rho_void) * xbar)

    The mass fraction must be well below the stiffness fraction.  At equal
    ratios a patch of void has the same ``E / rho`` as the plate, and soft
    regions with relatively heavy nodes vibrate on their own at low
    frequencies: spurious localised modes that would crowd the spectrum around
    the target (Pedersen, 2000).  With ``rho_void << e_void`` the void's own
    frequencies sit far above the plate's.
    """

    e: float
    rho: float
    e_void: float = 1e-3
    rho_void: float = 1e-5

    def youngs(self, xbar: np.ndarray) -> np.ndarray:
        return self.e * (self.e_void + (1.0 - self.e_void) * xbar)

    def density(self, xbar: np.ndarray) -> np.ndarray:
        return self.rho * (self.rho_void + (1.0 - self.rho_void) * xbar)

    def d_youngs(self, xbar: np.ndarray) -> np.ndarray:
        return np.full_like(xbar, self.e * (1.0 - self.e_void))

    def d_density(self, xbar: np.ndarray) -> np.ndarray:
        return np.full_like(xbar, self.rho * (1.0 - self.rho_void))
