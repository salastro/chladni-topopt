"""Method of Moving Asymptotes (Svanberg, 1987).

A self-contained implementation of the MMA subproblem and its primal-dual
interior-point solver, following Svanberg's formulation.  At each iteration the
true problem is replaced by the separable, convex approximation

    min  sum_j [ p0j/(U_j - x_j) + q0j/(x_j - L_j) ]  (+ artificial terms)
    s.t. sum_j [ pij/(U_j - x_j) + qij/(x_j - L_j) ] <= b_i

whose moving asymptotes ``L_j < x_j < U_j`` are pulled in when a variable
oscillates and pushed out when it moves steadily, which is what stabilises
problems like this one where the response is a non-monotonic function of the
design.

The subproblem carries the usual artificial variables ``y`` and ``z``::

    min f0(x) + a0*z + sum_i (c_i*y_i + d_i*y_i**2/2)
    s.t. f_i(x) - a_i*z - y_i <= 0,   y >= 0, z >= 0

With the defaults (``a0 = 1``, ``a_i = 0``, large ``c_i``) the artificial
variables are driven to zero and the original problem is recovered; they exist
so the subproblem is always feasible, even from an infeasible start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MMA:
    """Stateful MMA driver.

    Parameters
    ----------
    n, m:
        Number of design variables and constraints.
    xmin, xmax:
        Bounds, broadcast to ``(n,)``.
    move:
        Move limit as a fraction of ``xmax - xmin``.
    asyinit, asyincr, asydecr:
        Initial asymptote spread and the expansion/contraction factors applied
        when a variable moves monotonically or oscillates.
    c:
        Penalty on the artificial variables; must dominate the objective's
        scale or constraints will be traded away.
    """

    n: int
    m: int
    xmin: float | np.ndarray = 0.0
    xmax: float | np.ndarray = 1.0
    move: float = 0.5
    asyinit: float = 0.5
    asyincr: float = 1.2
    asydecr: float = 0.7
    albefa: float = 0.1
    raa0: float = 1e-5
    a0: float = 1.0
    a: np.ndarray | None = None
    c: float | np.ndarray = 1000.0
    d: float | np.ndarray = 1.0
    epsimin: float = 1e-7

    iteration: int = field(default=0, init=False)
    xold1: np.ndarray | None = field(default=None, init=False, repr=False)
    xold2: np.ndarray | None = field(default=None, init=False, repr=False)
    low: np.ndarray | None = field(default=None, init=False, repr=False)
    upp: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.xmin = np.broadcast_to(np.asarray(self.xmin, float), (self.n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(self.xmax, float), (self.n,)).copy()
        self.a = (
            np.zeros(self.m)
            if self.a is None
            else np.broadcast_to(np.asarray(self.a, float), (self.m,)).copy()
        )
        self.c = np.broadcast_to(np.asarray(self.c, float), (self.m,)).copy()
        self.d = np.broadcast_to(np.asarray(self.d, float), (self.m,)).copy()

    # -- asymptote update -----------------------------------------------

    def _update_asymptotes(self, x: np.ndarray) -> None:
        span = self.xmax - self.xmin
        if self.iteration <= 2:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
            return

        # Sign of the last two moves: oscillating -> pull asymptotes in.
        zzz = (x - self.xold1) * (self.xold1 - self.xold2)
        factor = np.ones(self.n)
        factor[zzz > 0.0] = self.asyincr
        factor[zzz < 0.0] = self.asydecr

        self.low = x - factor * (self.xold1 - self.low)
        self.upp = x + factor * (self.upp - self.xold1)
        self.low = np.clip(self.low, x - 10.0 * span, x - 0.01 * span)
        self.upp = np.clip(self.upp, x + 0.01 * span, x + 10.0 * span)

    # -- main entry point -------------------------------------------------

    def update(
        self,
        x: np.ndarray,
        f0: float,
        df0: np.ndarray,
        fi: np.ndarray,
        dfi: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, float)
        df0 = np.asarray(df0, float)
        fi = np.atleast_1d(np.asarray(fi, float))
        dfi = np.atleast_2d(np.asarray(dfi, float))

        self.iteration += 1
        self._update_asymptotes(x)

        span = np.maximum(self.xmax - self.xmin, 1e-5)
        alfa = np.maximum(
            np.maximum(self.xmin, self.low + self.albefa * (x - self.low)),
            x - self.move * span,
        )
        bta = np.minimum(
            np.minimum(self.xmax, self.upp - self.albefa * (self.upp - x)),
            x + self.move * span,
        )

        ux1 = self.upp - x
        xl1 = x - self.low
        ux2, xl2 = ux1**2, xl1**2

        p0 = np.maximum(df0, 0.0)
        q0 = np.maximum(-df0, 0.0)
        pq0 = 0.001 * (p0 + q0) + self.raa0 / span
        p0 = (p0 + pq0) * ux2
        q0 = (q0 + pq0) * xl2

        pij = np.maximum(dfi, 0.0)
        qij = np.maximum(-dfi, 0.0)
        pqij = 0.001 * (pij + qij) + self.raa0 / span
        pij = (pij + pqij) * ux2
        qij = (qij + pqij) * xl2

        b = pij @ (1.0 / ux1) + qij @ (1.0 / xl1) - fi

        xnew = _subsolve(
            m=self.m,
            n=self.n,
            epsimin=self.epsimin,
            low=self.low,
            upp=self.upp,
            alfa=alfa,
            bta=bta,
            p0=p0,
            q0=q0,
            pij=pij,
            qij=qij,
            a0=self.a0,
            a=self.a,
            b=b,
            c=self.c,
            d=self.d,
        )

        self.xold2 = self.xold1
        self.xold1 = x.copy()
        return xnew


def _subsolve(
    *,
    m: int,
    n: int,
    epsimin: float,
    low: np.ndarray,
    upp: np.ndarray,
    alfa: np.ndarray,
    bta: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    pij: np.ndarray,
    qij: np.ndarray,
    a0: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Primal-dual interior-point solve of one MMA subproblem.

    Follows the barrier parameter ``epsi`` down by factors of ten, taking
    damped Newton steps on the KKT residual at each level.
    """
    een, eem = np.ones(n), np.ones(m)

    x = 0.5 * (alfa + bta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (bta - x), een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()
    epsi = 1.0

    def residual(epsi_):
        ux1 = upp - x
        xl1 = x - low
        plam = p0 + pij.T @ lam
        qlam = q0 + qij.T @ lam
        gvec = pij @ (1.0 / ux1) + qij @ (1.0 / xl1)
        dpsidx = plam / ux1**2 - qlam / xl1**2
        return np.concatenate(
            [
                dpsidx - xsi + eta,
                c + d * y - mu - lam,
                np.array([a0 - zet - a @ lam]),
                gvec - a * z - y + s - b,
                xsi * (x - alfa) - epsi_,
                eta * (bta - x) - epsi_,
                mu * y - epsi_,
                np.array([zet * z - epsi_]),
                lam * s - epsi_,
            ]
        )

    while epsi > epsimin:
        res = residual(epsi)
        norm_res = np.linalg.norm(res)
        max_res = np.abs(res).max()

        for _ in range(200):
            if max_res <= 0.9 * epsi:
                break

            ux1 = upp - x
            xl1 = x - low
            ux2, xl2 = ux1**2, xl1**2
            ux3, xl3 = ux1 * ux2, xl1 * xl2

            plam = p0 + pij.T @ lam
            qlam = q0 + qij.T @ lam
            gvec = pij @ (1.0 / ux1) + qij @ (1.0 / xl1)
            gg = pij / ux2[None, :] - qij / xl2[None, :]
            dpsidx = plam / ux2 - qlam / xl2

            delx = dpsidx - epsi / (x - alfa) + epsi / (bta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam

            diagx = 2.0 * (plam / ux3 + qlam / xl3)
            diagx += xsi / (x - alfa) + eta / (bta - x)
            diagxinv = 1.0 / diagx
            diagy = d + mu / y
            diagyinv = 1.0 / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - gg @ (delx * diagxinv)
                rhs = np.concatenate([blam, [delz]])
                alam = np.diag(diaglamyi) + (gg * diagxinv[None, :]) @ gg.T
                mat = np.block(
                    [
                        [alam, a.reshape(-1, 1)],
                        [a.reshape(1, -1), np.array([[-zet / z]])],
                    ]
                )
                sol = np.linalg.solve(mat, rhs)
                dlam = sol[:m]
                dz = sol[m]
                dx = -delx * diagxinv - (gg.T @ dlam) * diagxinv
            else:
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                axx = np.diag(diagx) + (gg.T * diaglamyiinv[None, :]) @ gg
                azz = zet / z + a @ (a * diaglamyiinv)
                axz = -(gg.T @ (a * diaglamyiinv))
                bx = delx + gg.T @ (dellamyi * diaglamyiinv)
                bz = delz - a @ (dellamyi * diaglamyiinv)
                mat = np.block(
                    [[axx, axz.reshape(-1, 1)], [axz.reshape(1, -1), np.array([[azz]])]]
                )
                sol = np.linalg.solve(mat, -np.concatenate([bx, [bz]]))
                dx = sol[:n]
                dz = sol[n]
                dlam = (gg @ dx) * diaglamyiinv - dz * (a * diaglamyiinv)
                dlam += dellamyi * diaglamyiinv

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (bta - x) + (eta * dx) / (bta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam

            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])

            # Fraction-to-the-boundary rule, in both the primal box and the
            # non-negative variables.
            step = np.max(
                np.concatenate(
                    [
                        [-1.01 * (dxx / xx).min() if xx.size else 0.0],
                        [(-1.01 * dx / (x - alfa)).max()],
                        [(1.01 * dx / (bta - x)).max()],
                        [1.0],
                    ]
                )
            )
            step = 1.0 / step

            xo, yo, zo = x.copy(), y.copy(), z
            lamo, xsio, etao = lam.copy(), xsi.copy(), eta.copy()
            muo, zeto, so = mu.copy(), zet, s.copy()

            new_norm = 2.0 * norm_res
            for _ in range(50):
                x = xo + step * dx
                y = yo + step * dy
                z = zo + step * dz
                lam = lamo + step * dlam
                xsi = xsio + step * dxsi
                eta = etao + step * deta
                mu = muo + step * dmu
                zet = zeto + step * dzet
                s = so + step * ds
                res = residual(epsi)
                new_norm = np.linalg.norm(res)
                if new_norm <= norm_res:
                    break
                step *= 0.5

            norm_res = new_norm
            max_res = np.abs(res).max()

        epsi *= 0.1

    return x
