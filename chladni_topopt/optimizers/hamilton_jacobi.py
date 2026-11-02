"""Wang et al.'s level-set update: Steps 3-5 of their optimization algorithm.

The design variables are the nodal values of ``Phi`` on the
``(nx + 1) x (ny + 1)`` node grid.  Each update

1. combines objective and constraints into one sensitivity ``G``.  Wang et al.
   carry a single multiplier ``lambda_+`` for the volume (Eq. 32); the Chladni
   formulation has three constraints, so the augmented Lagrangian of
   :mod:`.lagrangian` supplies the multipliers instead.  ``G`` already holds
   the ``delta(Phi)`` factor of Eq. (28), because the adjoint chains through
   the smoothed Heaviside;
2. turns it into a normal speed ``V = -G`` (Eq. 28 with ``mu = 1``), optionally
   smoothed -- the non-local ``mu`` of Eq. (28) -- and scaled so its largest
   value is 1;
3. advances ``dPhi/dt = V |grad Phi|`` (Eq. 29) with the first-order upwind
   scheme of Osher and Sethian, Eq. (41), at a CFL-limited step, Eq. (42),
   with zero-flux edges, ``dPhi/dn = 0`` (Eq. 14).

Eq. (41) is printed without square roots on the gradient magnitudes; the
scheme it cites, and the one implemented, takes them.

Because ``V`` is normalised, the step moves the boundary by ``cfl`` element
widths at most, whatever the scale of the sensitivity -- the role ``lr``
plays for AdamW.  It decays geometrically for the same reason: at a constant
step the boundary keeps oscillating around the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lagrangian import AugmentedLagrangian


def upwind_step(phi: np.ndarray, speed: np.ndarray, dt: float, dx: float, dy: float) -> np.ndarray:
    """One explicit step of ``dPhi/dt = speed * |grad Phi|`` on a 2-D grid.

    Written as ``Phi_t + F |grad Phi| = 0`` with ``F = -speed``, the upwind
    scheme picks one-sided differences by the sign of ``F``.  Edges use
    mirrored ghost values, i.e. zero normal derivative.
    """
    p = np.pad(phi, 1, mode="edge")
    dxm = (p[1:-1, 1:-1] - p[:-2, 1:-1]) / dx
    dxp = (p[2:, 1:-1] - p[1:-1, 1:-1]) / dx
    dym = (p[1:-1, 1:-1] - p[1:-1, :-2]) / dy
    dyp = (p[1:-1, 2:] - p[1:-1, 1:-1]) / dy

    grad_plus = np.sqrt(
        np.maximum(dxm, 0.0) ** 2 + np.minimum(dxp, 0.0) ** 2
        + np.maximum(dym, 0.0) ** 2 + np.minimum(dyp, 0.0) ** 2
    )
    grad_minus = np.sqrt(
        np.maximum(dxp, 0.0) ** 2 + np.minimum(dxm, 0.0) ** 2
        + np.maximum(dyp, 0.0) ** 2 + np.minimum(dym, 0.0) ** 2
    )
    f = -speed
    return phi - dt * (np.maximum(f, 0.0) * grad_plus + np.minimum(f, 0.0) * grad_minus)


def smooth(field_2d: np.ndarray, passes: int) -> np.ndarray:
    """``passes`` rounds of 5-point averaging with zero-flux edges."""
    out = field_2d
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        out = (
            4.0 * p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        ) / 8.0
    return out


@dataclass
class HamiltonJacobi:
    """Upwind level-set evolution driven by the augmented-Lagrangian sensitivity.

    Parameters
    ----------
    grid:
        ``(nx + 1, ny + 1)``: the node grid ``Phi`` lives on, in the mesh's
        column-major node order.
    dx, dy:
        Node spacing.
    m:
        Number of inequality constraints.
    cfl:
        Largest boundary movement per update, in node spacings.  Above 1 the
        explicit scheme is unstable (Eq. 42), so each update is split into
        ``ceil(cfl)`` sub-steps.
    cfl_decay, cfl_min:
        Geometric decay per update, and its floor.
    smooth_passes:
        Rounds of neighbour averaging applied to the speed.  0 is Wang's
        ``mu = 1``; a few passes spread the speed over the band and damp
        node-to-node noise.
    rho, dual_interval:
        Augmented-Lagrangian penalty and dual-update interval.
    """

    grid: tuple[int, int]
    dx: float
    dy: float
    m: int
    cfl: float = 0.5
    cfl_decay: float = 0.996
    cfl_min: float = 0.05
    smooth_passes: int = 1
    rho: float = 10.0
    dual_interval: int = 5

    iteration: int = field(default=0, init=False)
    lagrangian: AugmentedLagrangian = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lagrangian = AugmentedLagrangian(
            m=self.m, rho=self.rho, dual_interval=self.dual_interval
        )

    @property
    def multipliers(self) -> np.ndarray:
        return self.lagrangian.multipliers

    @property
    def current_cfl(self) -> float:
        return max(self.cfl_min, self.cfl * self.cfl_decay**self.iteration)

    def update(
        self,
        x: np.ndarray,
        f0: float,
        df0: np.ndarray,
        fi: np.ndarray,
        dfi: np.ndarray,
    ) -> np.ndarray:
        fi = np.atleast_1d(np.asarray(fi, float))
        dfi = np.atleast_2d(np.asarray(dfi, float))
        cfl = self.current_cfl
        self.iteration += 1

        grad = self.lagrangian.gradient(np.asarray(df0, float), fi, dfi)
        speed = smooth(-grad.reshape(self.grid), self.smooth_passes)
        peak = np.max(np.abs(speed))

        phi = np.asarray(x, float).reshape(self.grid)
        if peak > 0.0:
            speed = speed / peak
            n_sub = int(np.ceil(cfl))
            dt = cfl * min(self.dx, self.dy) / n_sub
            for _ in range(n_sub):
                phi = upwind_step(phi, speed, dt, self.dx, self.dy)

        self.lagrangian.dual_step(fi, self.iteration)
        return phi.ravel()
