"""Time one optimization iteration, stage by stage, per solver configuration.

Runs a few real AdamW iterations of the Case 1 problem for each combination of
factorization backend and eigensolver setting, and reports where the time
goes.  This is the evidence behind the README's "Performance" section:

    python examples/benchmark_solvers.py                 # paper mesh, 100 x 120
    python examples/benchmark_solvers.py --quick         # 50 x 60
    python examples/benchmark_solvers.py --iterations 12

Nothing is written to disk.  Run it on an otherwise idle machine: the stages
are memory-bound and a concurrent job distorts them.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chladni_topopt.config import (  # noqa: E402
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
    build_problem,
)
from chladni_topopt.linalg import pardiso_available  # noqa: E402
from examples.shapes import letter_d  # noqa: E402


@dataclass
class StageTimer:
    """Wraps a callable and accumulates its wall time per call."""

    fn: object
    times: list[float] = field(default_factory=list)

    def __call__(self, *args, **kwargs):
        tic = time.perf_counter()
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.times.append(time.perf_counter() - tic)


def cpu_idle_fraction(seconds: float = 1.0) -> float:
    """Share of CPU time spent idle over a short window, from /proc/stat.

    The load average would also count this script's own earlier runs, so it
    reports "busy" on a machine that is actually free.
    """

    def sample():
        with open("/proc/stat") as f:
            fields = [float(v) for v in f.readline().split()[1:]]
        return fields[3] + fields[4], sum(fields)  # idle + iowait, total

    try:
        idle0, total0 = sample()
        time.sleep(seconds)
        idle1, total1 = sample()
    except OSError:
        return float("nan")
    return (idle1 - idle0) / max(total1 - total0, 1.0)


def run(
    solver: str,
    n_modes: int,
    warm: bool,
    iterations: int,
    quick: bool,
    limit_threads: bool = True,
) -> dict:
    config = ProblemConfig(
        targets=[TargetConfig(loops=letter_d(1.0, 1.2, 1.0, 1.2), mode_index=4)],
        mesh=MeshConfig(nx=100, ny=120),
        optimizer=OptimizerConfig(method="adamw", max_iterations=iterations),
        n_modes=n_modes,
        solver=solver,
    )
    if quick:
        config = config.coarsen(2)
    problem = build_problem(config, verbose=False)
    analysis = problem.analysis

    import chladni_topopt.eigen as eigen_module

    original_limit = eigen_module.openblas_single_threaded
    if not limit_threads:
        eigen_module.openblas_single_threaded = contextlib.nullcontext

    analysis.eigen_solver.n_modes = n_modes  # bypass the headroom guard's floor
    analysis.eigen_solver.warm_start = warm
    eig = StageTimer(analysis.eigen_solver.solve)
    analysis.eigen_solver.solve = eig
    grad = StageTimer(analysis.sensitivity.gradients)
    analysis.sensitivity.gradients = grad
    upd = StageTimer(problem.driver.optimizer.update)
    problem.driver.optimizer.update = upd

    tic = time.perf_counter()
    try:
        result = problem.run()
    finally:
        eigen_module.openblas_single_threaded = original_limit
    wall = time.perf_counter() - tic
    per_iter = [r.elapsed for r in result.history.records]

    # The first iteration pays one-off costs (a cold eigensolve, MKL start-up),
    # so it is excluded from the averages.
    skip = 1 if iterations > 2 else 0

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs[skip:])) if len(xs) > skip else float("nan")

    iteration = mean(per_iter) + mean(upd.times)
    return {
        "eigen": mean(eig.times),
        "adjoint": mean(grad.times),
        "update": mean(upd.times),
        "other": mean(per_iter) - mean(eig.times) - mean(grad.times),
        "iteration": iteration,
        "wall": wall,
        "final_freq": result.history.records[-1].target_frequencies[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    idle = cpu_idle_fraction()
    print(
        f"mesh {'50 x 60' if args.quick else '100 x 120'}, "
        f"{args.iterations} AdamW iterations per configuration, "
        f"CPU {100 * idle:.0f}% idle over 1 s before starting ({os.cpu_count()} CPUs)"
    )
    if idle < 0.85:
        print("WARNING: something else is using the CPU -- these timings are contended.")
    print()

    # (solver, modes, warm start, OpenBLAS limited).  The first row is the code
    # as it was before this work: SuperLU, 12 cold modes, default threading.
    configs = [
        ("superlu", 12, False, False),
        ("superlu", 12, False, True),
        ("superlu", 8, True, True),
    ]
    if pardiso_available():
        configs += [("pardiso", 12, False, True), ("pardiso", 8, True, True)]
    else:
        print("pypardiso not installed -- PARDISO rows skipped.\n")

    rows = []
    for solver, n_modes, warm, limited in configs:
        label = (
            f"{solver}, {n_modes} modes, {'warm' if warm else 'cold'}"
            + ("" if limited else ", all threads")
        )
        print(f"running {label} ...", flush=True)
        rows.append(
            (label, run(solver, n_modes, warm, args.iterations, args.quick, limited))
        )

    base = rows[0][1]["iteration"]
    header = f"{'configuration':38s}{'eigen':>9s}{'adjoint':>9s}{'update':>9s}{'other':>8s}{'per iter':>10s}{'speedup':>9s}"
    print("\n" + header + "\n" + "-" * len(header))
    for label, r in rows:
        print(
            f"{label:38s}{r['eigen']:9.3f}{r['adjoint']:9.3f}{r['update']:9.3f}"
            f"{r['other']:8.3f}{r['iteration']:10.3f}{base / r['iteration']:8.1f}x"
        )
    print("\nseconds per iteration, first iteration excluded; speedup vs the first row")
    freqs = {label: r["final_freq"] for label, r in rows}
    spread = max(freqs.values()) - min(freqs.values())
    print(f"final target frequency spread across configurations: {spread:.2e} Hz")


if __name__ == "__main__":
    main()
