"""Optimization formulations: Eq. (15) and the Appendix-A alternative, Eq. (21).

Both are written as a weighted sum over modes ``s`` so that the single-mode
case of Section 4.1 (one mode, ``w = 1``) and the multi-mode case of Section
4.2 share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .indicators import AVERAGE, VARIANCE
from .state import DesignState, Responses, TargetMode


@runtime_checkable
class Formulation(Protocol):
    """Turns a modal state into objective/constraint values and partials."""

    #: Names of the constraints, in order.
    constraint_names: tuple[str, ...]

    def responses(
        self, modes: Sequence[TargetMode], design: DesignState
    ) -> Responses:
        ...


def _volume_response(
    design: DesignState, vstar: float
) -> tuple[float, np.ndarray]:
    """Normalised volume constraint ``V / V* - 1 <= 0`` and its gradient.

    The paper writes ``V - V* <= 0``; dividing through by ``V*`` leaves the
    feasible set unchanged but puts the constraint on the same scale as the
    others, which is what MMA's subproblem wants.

    ``V*`` itself is never stated in the paper.  0.55 is inferred from Fig.
    5(a), where Case 1's volume fraction climbs from 0.50 and then runs flat at
    the reported 0.5498 -- the signature of a binding cap -- while Appendix B's
    falls to 0.4286 well inside it.
    """
    total = design.volumes.sum()
    value = design.volumes @ design.xbar / (total * vstar) - 1.0
    grad = design.volumes / (total * vstar)
    return float(value), grad


@dataclass(frozen=True)
class ConstraintFormulation:
    """Eq. (15) -- minimise the surrounding amplitude's reciprocal.

    ::

        min  C  = sum_s w_s Avg(L~_s, phi0_s) / Avg(L~_s, phi_s)
        s.t. g1 = sum_s w_s Avg(L_s, phi_s)  / Avg(L_s, phi0_s)   <= eps
             g2 = sum_s w_s Var(L~_s, phi_s) / Var(L~_s, phi0_s)  <= tau
             g3 = V - V* <= 0

    ``g1`` keeps the amplitude on the target curve small, which is what makes
    the nodal line appear there; ``g2`` keeps the surrounding amplitude
    uniform, which makes the pattern crisp; the objective pushes that
    surrounding amplitude up.

    ``vstar=None`` drops ``g3``.  A material budget made sense for the paper's
    two-material plate; for a single-material plate whose boundary is the
    design, it only fights the pattern constraints.
    """

    eps: float = 0.01
    tau: float = 0.4
    vstar: float | None = 0.55

    @property
    def constraint_names(self) -> tuple[str, ...]:
        names = ("avg_shape", "var_surround")
        return names if self.vstar is None else names + ("volume",)

    def responses(
        self, modes: Sequence[TargetMode], design: DesignState
    ) -> Responses:
        n_dof = len(modes[0].phi_full)
        n_modes = len(modes)
        n_resp = 1 + len(self.constraint_names)
        values = np.zeros(n_resp)
        dphi = np.zeros((n_resp, n_modes, n_dof))
        dlam = np.zeros((n_resp, n_modes))
        dxbar = np.zeros((n_resp, len(design.xbar)))

        obj = g_shape = g_var = 0.0
        for s, mode in enumerate(modes):
            sel, ref, w = mode.selection, mode.reference, mode.weight
            phi = mode.phi_full

            a_sur = AVERAGE.value(sel.surround, phi)
            a_shp = AVERAGE.value(sel.shape, phi)
            v_sur = VARIANCE.value(sel.surround, phi)

            obj += w * ref.avg_surround / a_sur
            g_shape += w * a_shp / ref.avg_shape
            g_var += w * v_sur / ref.var_surround

            dphi[0, s] = -w * ref.avg_surround / a_sur**2 * AVERAGE.gradient(
                sel.surround, phi
            )
            dphi[1, s] = w / ref.avg_shape * AVERAGE.gradient(sel.shape, phi)
            dphi[2, s] = w / ref.var_surround * VARIANCE.gradient(sel.surround, phi)

        values[0] = obj
        values[1] = g_shape - self.eps
        values[2] = g_var - self.tau
        if self.vstar is not None:
            values[3], dxbar[3] = _volume_response(design, self.vstar)

        diagnostics = {
            "obj": obj,
            "avg_shape_ratio": g_shape,
            "avg_surround_ratio": 1.0 / obj if obj else np.inf,
            "var_surround_ratio": g_var,
            "volume_fraction": design.volume_fraction,
        }
        return Responses(
            names=("objective",) + self.constraint_names,
            values=values,
            dphi=dphi,
            dlam=dlam,
            dxbar=dxbar,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class MultiObjectiveFormulation:
    """Eq. (21) -- the three indicators folded into one weighted objective.

    ::

        min  C = alpha sum_s w_s Avg(L_s, phi_s)  / Avg(L_s, phi0_s)
               + beta  sum_s w_s Avg(L~_s, phi0_s)/ Avg(L~_s, phi_s)
               + gamma sum_s w_s Var(L~_s, phi_s) / Var(L~_s, phi0_s)
        s.t. V - V* <= 0

    With ``alpha + beta + gamma = 1`` the objective starts at exactly 1, as the
    paper notes.  Appendix B uses ``(0.9801, 0.0099, 0.01)`` for the same
    single-mode letter-D problem.
    """

    alpha: float = 0.9801
    beta: float = 0.0099
    gamma: float = 0.01
    vstar: float = 0.55

    constraint_names: tuple[str, ...] = ("volume",)

    def __post_init__(self) -> None:
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"alpha + beta + gamma must be 1 for the objective to start at 1, got {total}"
            )

    def responses(
        self, modes: Sequence[TargetMode], design: DesignState
    ) -> Responses:
        n_dof = len(modes[0].phi_full)
        n_modes = len(modes)
        values = np.zeros(2)
        dphi = np.zeros((2, n_modes, n_dof))
        dlam = np.zeros((2, n_modes))
        dxbar = np.zeros((2, len(design.xbar)))

        obj = 0.0
        terms = np.zeros(3)
        for s, mode in enumerate(modes):
            sel, ref, w = mode.selection, mode.reference, mode.weight
            phi = mode.phi_full

            a_shp = AVERAGE.value(sel.shape, phi)
            a_sur = AVERAGE.value(sel.surround, phi)
            v_sur = VARIANCE.value(sel.surround, phi)

            t = np.array(
                [
                    a_shp / ref.avg_shape,
                    ref.avg_surround / a_sur,
                    v_sur / ref.var_surround,
                ]
            )
            terms += w * t
            obj += w * (self.alpha * t[0] + self.beta * t[1] + self.gamma * t[2])

            dphi[0, s] = w * (
                self.alpha / ref.avg_shape * AVERAGE.gradient(sel.shape, phi)
                - self.beta
                * ref.avg_surround
                / a_sur**2
                * AVERAGE.gradient(sel.surround, phi)
                + self.gamma
                / ref.var_surround
                * VARIANCE.gradient(sel.surround, phi)
            )

        vol, dvol = _volume_response(design, self.vstar)
        values[0] = obj
        values[1] = vol
        dxbar[1] = dvol

        diagnostics = {
            "obj": obj,
            "avg_shape_ratio": float(terms[0]),
            "avg_surround_ratio": 1.0 / float(terms[1]) if terms[1] else np.inf,
            "var_surround_ratio": float(terms[2]),
            "volume_fraction": design.volume_fraction,
        }
        return Responses(
            names=("objective",) + self.constraint_names,
            values=values,
            dphi=dphi,
            dlam=dlam,
            dxbar=dxbar,
            diagnostics=diagnostics,
        )


@dataclass
class SignedPatternFormulation:
    """A loss that only a real nodal line along the target can satisfy.

    The paper's indicators compare each amplitude with its value for the
    initial design.  An optimizer can then meet ``Avg(L, phi)`` by making the
    whole neighbourhood of the target quiet -- pushing the mode onto a thin
    flap or up the spectrum -- without any line forming.  In this repository
    that is how most failed runs failed.  Every term here is unchanged by
    scaling the mode or flipping its sign, and together they leave no quiet
    way out::

        min  C  = r / r0,           r = Avg(L, phi) / Avg(L~, phi)
        s.t. g1 = (1 - c) - kappa <= 0
             g2 = eta_min - eta / eta0 <= 0
             g3 = V - V* <= 0        (only if vstar is set)

    * ``r`` is the band amplitude *relative to* the surround.  A quiet
      neighbourhood lowers both, so it does not help.
    * ``c = (sum_i s_i phi_i)^2 / (sum_i |phi_i|)^2`` over the surround
      nodes, where ``s_i = +-1`` is the node's side of the target.  It is 1
      exactly when one side moves one way and the other side the opposite
      way, which puts a nodal line between them, in the band or the gap.
    * ``eta`` is the share of the mode's transverse motion, ``sum phi_i^2``,
      that lies in the controlled zone (band plus surround).  Keeping it above
      ``eta_min`` times its initial value stops the mode escaping to a flap.

    ``|phi_i|`` is smoothed as ``sqrt(phi_i^2 + delta^2)``; ``delta`` is tiny
    against the amplitudes of an M-normalised mode.  ``r0`` is the initial
    design's ratio, from the reference indicators; ``eta0`` is recorded on the
    first call.  Mode weights ``w_s`` combine targets as in Eq. (15).
    """

    kappa: float = 0.05
    eta_min: float = 0.5
    vstar: float | None = None
    delta: float = 1e-9
    eta_ref: list[float] | None = None

    @property
    def constraint_names(self) -> tuple[str, ...]:
        names = ("sign_mismatch", "zone_energy")
        return names if self.vstar is None else names + ("volume",)

    @staticmethod
    def _layout(sel, n_dof: int):
        """Side of each surround DOF, and masks of the zone and of all transverse DOFs."""
        ndof = n_dof // len(sel.signed_distance)
        sur = np.flatnonzero(sel.surround)
        transverse = sur[0] % ndof
        side = np.sign(sel.signed_distance[sur // ndof])
        zone = (sel.shape + sel.surround) > 0
        trans = np.zeros(n_dof, dtype=bool)
        trans[transverse::ndof] = True
        return sur, side, zone, trans

    def responses(
        self, modes: Sequence[TargetMode], design: DesignState
    ) -> Responses:
        n_dof = len(modes[0].phi_full)
        n_modes = len(modes)
        n_resp = 1 + len(self.constraint_names)
        values = np.zeros(n_resp)
        dphi = np.zeros((n_resp, n_modes, n_dof))
        dlam = np.zeros((n_resp, n_modes))
        dxbar = np.zeros((n_resp, len(design.xbar)))
        first = self.eta_ref is None
        if first:
            self.eta_ref = []

        obj = mismatch = zone_short = 0.0
        shape_ratio = var_ratio = sign_agree = eta_ratio = 0.0
        for s, mode in enumerate(modes):
            sel, ref, w = mode.selection, mode.reference, mode.weight
            phi = mode.phi_full
            sur, side, zone, trans = self._layout(sel, n_dof)

            # Band ratio r = Avg(L) / Avg(L~), against the initial design's.
            a_shp = AVERAGE.value(sel.shape, phi)
            a_sur = AVERAGE.value(sel.surround, phi)
            r0 = ref.avg_shape / ref.avg_surround
            r = a_shp / a_sur
            obj += w * r / r0
            dphi[0, s] = w / r0 * (
                AVERAGE.gradient(sel.shape, phi) / a_sur
                - a_shp / a_sur**2 * AVERAGE.gradient(sel.surround, phi)
            )

            # Sign agreement c = S^2 / A^2 across the target.
            p = phi[sur]
            mag = np.sqrt(p**2 + self.delta**2)
            S, A = side @ p, mag.sum()
            c = S**2 / A**2
            mismatch += w * (1.0 - c)
            g = np.zeros(n_dof)
            g[sur] = -(2.0 * S / A**2 * side - 2.0 * S**2 / A**3 * p / mag)
            dphi[1, s] = w * g

            # Share of the transverse motion inside the controlled zone.
            Z = float(phi[zone] @ phi[zone])
            T = float(phi[trans] @ phi[trans])
            eta = Z / T
            if first:
                self.eta_ref.append(eta)
            eta0 = self.eta_ref[s]
            zone_short += w * (self.eta_min - eta / eta0)
            g = np.zeros(n_dof)
            g[trans] = -2.0 * phi[trans] * (zone[trans] * T - Z) / (T**2 * eta0)
            dphi[2, s] = w * g

            shape_ratio += w * a_shp / ref.avg_shape
            var_ratio += w * VARIANCE.value(sel.surround, phi) / ref.var_surround
            sign_agree += w * c
            eta_ratio += w * eta / eta0

        values[0] = obj
        values[1] = mismatch - self.kappa
        values[2] = zone_short
        if self.vstar is not None:
            values[3], dxbar[3] = _volume_response(design, self.vstar)

        diagnostics = {
            "obj": obj,
            "band_ratio": obj,
            "sign_agreement": sign_agree,
            "zone_energy_ratio": eta_ratio,
            # The paper's indicators, for plots and comparison with Eq. (15) runs.
            "avg_shape_ratio": shape_ratio,
            "avg_surround_ratio": shape_ratio / obj if obj else np.inf,
            "var_surround_ratio": var_ratio,
            "volume_fraction": design.volume_fraction,
        }
        return Responses(
            names=("objective",) + self.constraint_names,
            values=values,
            dphi=dphi,
            dlam=dlam,
            dxbar=dxbar,
            diagnostics=diagnostics,
        )
