"""The optimization loop of Fig. 3.

    initialise -> analyse -> track modes -> responses -> adjoint sensitivities
    -> filter/project -> design update -> converged? -> stop

The driver holds no numerics of its own: every stage is an injected component,
and it only sequences them, records history and decides when to stop.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .analysis import ChladniAnalysis, ModalState
from .filters import Continuation
from .optimizers.base import Optimizer
from .state import Responses


@dataclass
class WindowedConvergence:
    """Eq. (20): compare the last ``window`` objectives with the ``window`` before.

    ::

        error = |sum_{n=0..4} (Obj_{it-n} - Obj_{it-n-5})| / sum_{n=0..4} Obj_{it-n}

    (The paper prints the second index as ``it-(n-5)``, which would reference
    future iterations; this is the standard backward-looking form.)

    ``min_since_reset`` keeps the test from firing on the flat stretch that
    immediately follows a beta update, when the history still describes the
    previous projection sharpness.

    The paper's criterion looks at the objective alone.  That is not enough
    here: the objective plateaus early while the eigenvector constraints are
    still far from satisfied, and acting on it would escalate beta -- and
    eventually stop the run -- at an infeasible design.  Convergence therefore
    also requires the constraints to be met to within ``feasibility_tol``.
    """

    window: int = 5
    tol: float = 0.002
    feasibility_tol: float = 1e-3
    min_since_reset: int = 0
    _history: deque[float] = field(default_factory=deque, init=False)
    _since_reset: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.min_since_reset == 0:
            self.min_since_reset = 2 * self.window
        self._history = deque(maxlen=2 * self.window)

    def reset(self) -> None:
        self._since_reset = 0

    def push(self, objective: float, constraints: np.ndarray | None = None) -> float:
        """Record an objective value and return the current error measure.

        Returns ``inf`` while the design is infeasible or the window is not yet
        full, so the caller can treat "error <= tol" as "settled".
        """
        self._history.append(float(objective))
        if constraints is not None and len(constraints):
            if float(np.max(constraints)) > self.feasibility_tol:
                self._since_reset += 1
                return np.inf
        self._since_reset += 1
        if len(self._history) < 2 * self.window or self._since_reset < self.min_since_reset:
            return np.inf
        recent = np.array(self._history)[-self.window :]
        previous = np.array(self._history)[: self.window]
        denom = recent.sum()
        if denom == 0.0:
            return np.inf
        return float(abs((recent - previous).sum()) / abs(denom))


@dataclass
class IterationRecord:
    """One row of the optimization history."""

    iteration: int
    objective: float
    constraints: np.ndarray
    frequencies: np.ndarray
    target_indices: tuple[int, ...]
    target_frequencies: tuple[float, ...]
    macs: tuple[float, ...]
    switched: bool
    beta: float
    error: float
    diagnostics: dict[str, float]
    elapsed: float

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "objective": self.objective,
            "constraints": self.constraints.tolist(),
            "frequencies": self.frequencies.tolist(),
            "target_indices": list(self.target_indices),
            "target_frequencies": list(self.target_frequencies),
            "macs": list(self.macs),
            "switched": self.switched,
            "beta": self.beta if np.isfinite(self.beta) else None,
            "error": None if not np.isfinite(self.error) else self.error,
            "diagnostics": dict(self.diagnostics),
            "elapsed": self.elapsed,
        }


@dataclass
class History:
    """Accumulated per-iteration records, with convenience accessors."""

    records: list[IterationRecord] = field(default_factory=list)

    def append(self, record: IterationRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def objective(self) -> np.ndarray:
        return np.array([r.objective for r in self.records])

    @property
    def beta(self) -> np.ndarray:
        return np.array([r.beta for r in self.records])

    @property
    def frequencies(self) -> np.ndarray:
        """``(n_iter, n_modes_computed)`` frequency history of the whole spectrum."""
        return np.array([r.frequencies for r in self.records])

    @property
    def target_frequencies(self) -> np.ndarray:
        return np.array([r.target_frequencies for r in self.records])

    def diagnostic(self, key: str) -> np.ndarray:
        return np.array([r.diagnostics[key] for r in self.records])

    def to_dict(self) -> dict:
        return {"records": [r.to_dict() for r in self.records]}


@dataclass
class OptimizationResult:
    """Final design plus everything needed to plot and re-analyse it."""

    x: np.ndarray
    state: ModalState
    responses: Responses
    history: History
    converged: bool
    iterations: int


Callback = Callable[[int, ModalState, Responses, "History"], None]


@dataclass
class TopologyOptimizer:
    """Sequences analysis, sensitivity, continuation and the design update."""

    analysis: ChladniAnalysis
    optimizer: Optimizer
    continuation: Continuation
    convergence: WindowedConvergence = field(default_factory=WindowedConvergence)
    max_iterations: int = 600
    callbacks: Sequence[Callback] = ()
    verbose: bool = True

    def run(self, x0: np.ndarray) -> OptimizationResult:
        x = np.array(x0, dtype=float)
        history = History()
        state = responses = None
        converged = False

        for it in range(1, self.max_iterations + 1):
            tic = time.perf_counter()
            state, responses, gradients = self.analysis.evaluate(x)
            error = self.convergence.push(
                responses.objective, responses.constraints
            )

            # The run may only stop once the projection has reached its final
            # sharpness; before that, convergence just advances the schedule.
            settled = error <= self.convergence.tol
            at_max = self.continuation.at_max
            if self.continuation.step(settled):
                self.convergence.reset()

            record = IterationRecord(
                iteration=it,
                objective=responses.objective,
                constraints=responses.constraints.copy(),
                frequencies=state.solution.frequencies.copy(),
                target_indices=tuple(m.index for m in state.modes),
                target_frequencies=tuple(m.frequency for m in state.modes),
                macs=tuple(t.mac for t in state.tracking),
                switched=any(t.switched for t in state.tracking),
                beta=self.continuation.beta,
                error=error,
                diagnostics=dict(responses.diagnostics),
                elapsed=time.perf_counter() - tic,
            )
            history.append(record)
            if self.verbose:
                self._log(record)
            for callback in self.callbacks:
                callback(it, state, responses, history)

            if settled and at_max:
                converged = True
                break

            x = self.optimizer.update(
                x,
                responses.objective,
                gradients[0],
                responses.constraints,
                gradients[1:],
            )

        return OptimizationResult(
            x=x,
            state=state,
            responses=responses,
            history=history,
            converged=converged,
            iterations=len(history),
        )

    @staticmethod
    def _log(record: IterationRecord) -> None:
        cons = " ".join(f"{c:+.3e}" for c in record.constraints)
        freqs = " ".join(f"{f:7.2f}" for f in record.target_frequencies)
        err = "   --  " if not np.isfinite(record.error) else f"{record.error:.1e}"
        switched = " *switch*" if record.switched else ""
        print(
            f"it {record.iteration:4d} | obj {record.objective:10.5f} | "
            f"g [{cons}] | vol {record.diagnostics['volume_fraction']:.4f} | "
            f"f {freqs} Hz | mode {list(record.target_indices)} | "
            f"beta {record.beta:5.1f} | err {err}{switched}",
            flush=True,
        )
