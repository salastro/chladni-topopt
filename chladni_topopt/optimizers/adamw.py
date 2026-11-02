"""AdamW with augmented-Lagrangian constraint handling.

AdamW (Loshchilov & Hutter, 2019) is an unconstrained, first-order method, and
the problems here are not unconstrained: Eq. (15) carries three inequality
constraints, and every design variable is boxed to ``[0, 1]``.  This wrapper
supplies both:

* **Bounds** by projection -- each step is clipped back into ``[xmin, xmax]``.
* **Inequalities** by a Powell-Hestenes-Rockafellar augmented Lagrangian
  (:class:`~chladni_topopt.optimizers.lagrangian.AugmentedLagrangian`): Adam
  descends on its gradient, and the multipliers take a dual ascent step every
  ``dual_interval`` iterations.

Adam divides each coordinate's step by a running RMS of its gradient, so ``lr``
is roughly the largest change any design variable makes per iteration -- the
same role MMA's move limit plays.  The same normalisation means the step does
*not* shrink near a solution: at a constant rate the iterates keep circling the
optimum at a radius of about ``lr``, and the multipliers circle with them.  The
rate therefore decays geometrically, ``lr_t = max(lr_min, lr * lr_decay**t)``.

Two things behave differently from MMA and matter for this problem.  The
augmented Lagrangian only reaches feasibility as the multipliers grow, so the
early iterates can sit well outside the constraints; and the weight decay of
AdamW pulls every variable towards ``decay_target``.  The textbook target is 0,
which here would mean "towards epoxy" -- a bias with no physical reason -- so
the default decay is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .lagrangian import AugmentedLagrangian


@dataclass
class AdamW:
    """Projected AdamW on an augmented Lagrangian.

    Parameters
    ----------
    n, m:
        Number of design variables and inequality constraints.
    xmin, xmax:
        Box bounds, enforced by clipping after every step.
    lr:
        Initial step size.  Roughly the per-iteration change of a variable.
    lr_decay, lr_min:
        Geometric decay factor per iteration, and the floor it stops at.  The
        defaults take 0.02 down to 0.002 over about 600 iterations.
    beta1, beta2, eps:
        Adam's moment decay rates and denominator guard.
    weight_decay:
        Decoupled decay coefficient.  0 by default; see the module docstring.
    decay_target:
        Point the decay pulls towards.  0 gives textbook AdamW.
    rho:
        Augmented-Lagrangian penalty.  The constraints are normalised to O(1),
        so moderate values work; too large and the problem becomes as stiff as
        a pure penalty method.
    dual_interval:
        Primal steps between multiplier updates.
    """

    n: int
    m: int
    xmin: float | np.ndarray = 0.0
    xmax: float | np.ndarray = 1.0
    lr: float = 0.02
    lr_decay: float = 0.9962
    lr_min: float = 0.002
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    decay_target: float = 0.0
    rho: float = 10.0
    dual_interval: int = 5

    iteration: int = field(default=0, init=False)
    lagrangian: AugmentedLagrangian = field(init=False, repr=False)
    _m1: np.ndarray = field(init=False, repr=False)
    _m2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.xmin = np.broadcast_to(np.asarray(self.xmin, float), (self.n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(self.xmax, float), (self.n,)).copy()
        self.lagrangian = AugmentedLagrangian(
            m=self.m, rho=self.rho, dual_interval=self.dual_interval
        )
        self._m1 = np.zeros(self.n)
        self._m2 = np.zeros(self.n)

    @property
    def current_lr(self) -> float:
        """Step size the next update will use."""
        return max(self.lr_min, self.lr * self.lr_decay**self.iteration)

    @property
    def multipliers(self) -> np.ndarray:
        return self.lagrangian.multipliers

    def lagrangian_gradient(
        self, df0: np.ndarray, fi: np.ndarray, dfi: np.ndarray
    ) -> np.ndarray:
        """Gradient of the augmented Lagrangian at the current multipliers."""
        return self.lagrangian.gradient(df0, fi, dfi)

    def update(
        self,
        x: np.ndarray,
        f0: float,
        df0: np.ndarray,
        fi: np.ndarray,
        dfi: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, float)
        fi = np.atleast_1d(np.asarray(fi, float))
        dfi = np.atleast_2d(np.asarray(dfi, float))
        lr = self.current_lr
        self.iteration += 1
        t = self.iteration

        grad = self.lagrangian_gradient(np.asarray(df0, float), fi, dfi)

        self._m1 = self.beta1 * self._m1 + (1.0 - self.beta1) * grad
        self._m2 = self.beta2 * self._m2 + (1.0 - self.beta2) * grad**2
        m_hat = self._m1 / (1.0 - self.beta1**t)
        v_hat = self._m2 / (1.0 - self.beta2**t)

        step = m_hat / (np.sqrt(v_hat) + self.eps)
        decay = self.weight_decay * (x - self.decay_target)
        xnew = np.clip(x - lr * (step + decay), self.xmin, self.xmax)

        self.lagrangian.dual_step(fi, t)

        return xnew
