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
    """

    eps: float = 0.01
    tau: float = 0.4
    vstar: float = 0.55

    constraint_names: tuple[str, ...] = ("avg_shape", "var_surround", "volume")

    def responses(
        self, modes: Sequence[TargetMode], design: DesignState
    ) -> Responses:
        n_dof = len(modes[0].phi_full)
        n_modes = len(modes)
        values = np.zeros(4)
        dphi = np.zeros((4, n_modes, n_dof))
        dlam = np.zeros((4, n_modes))
        dxbar = np.zeros((4, len(design.xbar)))

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

        vol, dvol = _volume_response(design, self.vstar)

        values[0] = obj
        values[1] = g_shape - self.eps
        values[2] = g_var - self.tau
        values[3] = vol
        dxbar[3] = dvol

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
