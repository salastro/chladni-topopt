"""Convergence bookkeeping and an end-to-end run of the whole pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.config import (
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
    build_problem,
)
from chladni_topopt.driver import TopologyOptimizer, WindowedConvergence
from chladni_topopt.filters import BetaContinuation, ExponentialThreshold
from chladni_topopt.formulations import ConstraintFormulation
from examples.shapes import letter_d

FEASIBLE = np.array([-1.0, -1.0, -1.0])
INFEASIBLE = np.array([0.5, -1.0, -1.0])


def test_convergence_needs_a_full_window():
    c = WindowedConvergence(window=5, tol=0.002, min_since_reset=10)
    for _ in range(9):
        assert not np.isfinite(c.push(1.0, FEASIBLE))
    assert c.push(1.0, FEASIBLE) == pytest.approx(0.0)


def test_convergence_measures_drift_between_windows():
    c = WindowedConvergence(window=5, tol=0.002, min_since_reset=10)
    for _ in range(5):
        c.push(1.0, FEASIBLE)
    for _ in range(4):
        c.push(1.1, FEASIBLE)
    error = c.push(1.1, FEASIBLE)
    # Five values moved from 1.0 to 1.1 against a window sum of 5.5.
    assert error == pytest.approx(0.5 / 5.5, rel=1e-9)


def test_infeasible_designs_never_count_as_converged():
    """A plateaued objective at a violated constraint is not convergence."""
    c = WindowedConvergence(window=5, tol=0.002, min_since_reset=10)
    for _ in range(30):
        assert not np.isfinite(c.push(1.0, INFEASIBLE))
    # Once feasible, the same flat history converges immediately.
    for _ in range(10):
        c.push(1.0, FEASIBLE)
    assert c.push(1.0, FEASIBLE) == pytest.approx(0.0)


def test_reset_blocks_convergence_until_the_window_refills():
    c = WindowedConvergence(window=5, tol=0.002, min_since_reset=10)
    for _ in range(10):
        c.push(1.0, FEASIBLE)
    assert c.push(1.0, FEASIBLE) == pytest.approx(0.0)
    c.reset()
    assert not np.isfinite(c.push(1.0, FEASIBLE))


def test_run_terminates_only_at_maximum_beta(analysis, mesh):
    """Convergence below beta_max advances the schedule instead of stopping.

    The feasibility gate is switched off here: the frozen design sits at the
    uniform start, where the constraints are of course violated, and this test
    is about the beta schedule rather than feasibility (covered above).
    """
    projection = analysis.projection
    continuation = BetaContinuation(
        projection=projection, beta0=1.0, beta_max=4.0, interval=2
    )
    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=_FrozenOptimizer(),
        continuation=continuation,
        convergence=WindowedConvergence(
            window=2, tol=1.0, min_since_reset=2, feasibility_tol=np.inf
        ),
        max_iterations=12,
        verbose=False,
    )
    result = driver.run(np.full(mesh.n_elem, 0.5))
    assert projection.beta == 4.0
    assert result.converged
    # It cannot have stopped on the very first convergence, at beta = 1.
    assert result.iterations > 2


class _FrozenOptimizer:
    """An optimizer that never moves, so only the schedule drives the loop."""

    def update(self, x, f0, df0, fi, dfi):
        return x


@pytest.mark.slow
def test_end_to_end_improves_the_pattern():
    """A short run on a coarse mesh must reduce the amplitude on the target.

    Not a reproduction of the paper's result -- it is far too short for that --
    but it exercises assembly, eigensolve, tracking, responses, the adjoint,
    continuation and MMA together, and asserts the one thing the method is for.
    """
    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=100, ny=120),
        optimizer=OptimizerConfig(max_iterations=25),
        formulation=ConstraintFormulation(),
        n_modes=8,
    ).coarsen(4)

    problem = build_problem(config, verbose=False)
    result = problem.run()
    history = result.history

    assert history.diagnostic("avg_shape_ratio")[0] == pytest.approx(1.0)
    assert history.diagnostic("var_surround_ratio")[0] == pytest.approx(1.0)
    assert history.objective[0] == pytest.approx(1.0)

    # The point of the method: less amplitude on the target curve than before.
    assert history.diagnostic("avg_shape_ratio")[-1] < 0.5
    assert np.all(np.isfinite(history.objective))
    assert result.state.design.xbar.min() >= -1e-12
    assert result.state.design.xbar.max() <= 1.0 + 1e-12


def test_history_serialises_round_trip(tmp_path, analysis, mesh):
    from chladni_topopt.io import load_history, save_history

    driver = TopologyOptimizer(
        analysis=analysis,
        optimizer=_FrozenOptimizer(),
        continuation=BetaContinuation(
            projection=ExponentialThreshold(eta=0.5, beta=1.0), beta_max=1.0
        ),
        convergence=WindowedConvergence(
            window=2, tol=1.0, min_since_reset=2, feasibility_tol=np.inf
        ),
        max_iterations=4,
        verbose=False,
    )
    result = driver.run(np.full(mesh.n_elem, 0.5))
    path = save_history(tmp_path / "history.json", result.history)
    data = load_history(path)
    assert len(data["records"]) == len(result.history)
    assert data["records"][0]["objective"] == pytest.approx(1.0)
