"""MMA against problems with known optima."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.optimizers.mma import MMA

# Svanberg's five-variable cantilever benchmark:
#   min 0.0624 * sum(x)
#   s.t. 61/x1^3 + 37/x2^3 + 19/x3^3 + 7/x4^3 + 1/x5^3 <= 1
# with the published optimum f0* = 1.3399 at x* = (6.016, 5.309, 4.494, 3.502, 2.153).
CANTILEVER_COEF = np.array([61.0, 37.0, 19.0, 7.0, 1.0])
CANTILEVER_OPTIMUM = np.array([6.016, 5.309, 4.494, 3.502, 2.153])


def _cantilever(x):
    f0 = 0.0624 * x.sum()
    df0 = np.full(5, 0.0624)
    fi = np.array([np.sum(CANTILEVER_COEF / x**3) - 1.0])
    dfi = (-3.0 * CANTILEVER_COEF / x**4).reshape(1, 5)
    return f0, df0, fi, dfi


def test_cantilever_benchmark():
    """Reproduces Svanberg's published optimum for his own test problem."""
    x = np.full(5, 5.0)
    opt = MMA(n=5, m=1, xmin=1e-3, xmax=10.0, move=1.0)
    for _ in range(60):
        x = opt.update(x, *_cantilever(x))

    f0, _, fi, _ = _cantilever(x)
    assert f0 == pytest.approx(1.3399, abs=1e-3)
    assert np.max(np.abs(x - CANTILEVER_OPTIMUM)) < 5e-3
    assert fi[0] <= 1e-6


@pytest.mark.parametrize("x0", [2.0, 8.0])
def test_cantilever_reaches_the_same_optimum_from_any_start(x0):
    """The artificial variables keep the subproblem solvable when x0 violates g.

    From ``x0 = 2`` the start is deeply infeasible (g = 14.6) and the recovery
    costs roughly 100 iterations; from ``x0 = 8`` it is feasible but far out.
    """
    x = np.full(5, x0)
    opt = MMA(n=5, m=1, xmin=1e-3, xmax=10.0, move=1.0)
    for _ in range(120):
        x = opt.update(x, *_cantilever(x))
    f0, _, fi, _ = _cantilever(x)
    assert fi[0] <= 1e-6
    assert f0 == pytest.approx(1.3399, abs=1e-3)
    assert np.max(np.abs(x - CANTILEVER_OPTIMUM)) < 5e-3


def test_separable_quadratic_with_active_constraint():
    """Known KKT point: min ||x||^2 s.t. sum(x) >= 1 gives x_j = 1/n."""
    n = 6
    x = np.full(n, 0.9)

    def f(x):
        return x @ x, 2.0 * x, np.array([1.0 - x.sum()]), -np.ones((1, n))

    opt = MMA(n=n, m=1, xmin=0.0, xmax=2.0, move=0.3)
    for _ in range(80):
        x = opt.update(x, *f(x))
    assert np.allclose(x, 1.0 / n, atol=1e-4)


def test_more_constraints_than_variables_uses_the_other_branch():
    """Exercises the ``m >= n`` path of the subproblem solver.

    min x1 + x2 subject to three half-planes whose binding pair meets at
    (1, 1) -- a different linear-algebra branch from the topology problem,
    which always has far more variables than constraints.
    """
    def f(x):
        f0 = x.sum()
        df0 = np.ones(2)
        fi = np.array([1.0 - x[0], 1.0 - x[1], 1.5 - x[0] - x[1]])
        dfi = np.array([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]])
        return f0, df0, fi, dfi

    x = np.full(2, 3.0)
    opt = MMA(n=2, m=3, xmin=0.0, xmax=5.0, move=0.5)
    for _ in range(80):
        x = opt.update(x, *f(x))
    assert np.allclose(x, [1.0, 1.0], atol=1e-4)


def test_respects_bounds_and_move_limits():
    """Every iterate stays inside the box and inside the move limit."""
    n = 20
    rng = np.random.default_rng(0)
    target = rng.uniform(0.0, 1.0, n)
    x = np.full(n, 0.5)
    move = 0.1
    opt = MMA(n=n, m=1, xmin=0.0, xmax=1.0, move=move)

    for _ in range(30):
        f0 = float(((x - target) ** 2).sum())
        df0 = 2.0 * (x - target)
        fi = np.array([x.sum() / n - 0.9])
        dfi = np.full((1, n), 1.0 / n)
        xnew = opt.update(x, f0, df0, fi, dfi)
        assert np.all(xnew >= -1e-12) and np.all(xnew <= 1.0 + 1e-12)
        assert np.max(np.abs(xnew - x)) <= move + 1e-9
        x = xnew


def test_asymptotes_contract_when_a_variable_oscillates():
    """An oscillating variable must get a narrower trust region, not a wider one."""
    opt = MMA(n=2, m=1, xmin=0.0, xmax=1.0)
    opt.iteration = 3
    opt.xold1 = np.array([0.5, 0.5])
    opt.xold2 = np.array([0.4, 0.6])
    opt.low = np.array([0.0, 0.0])
    opt.upp = np.array([1.0, 1.0])
    # Variable 0 keeps rising (0.4 -> 0.5 -> 0.6); variable 1 reverses.
    opt._update_asymptotes(np.array([0.6, 0.6]))
    width = opt.upp - opt.low
    assert width[0] > width[1]
