"""Periodic figures and snapshots while an optimization is still running.

A paper-resolution run takes the better part of an hour.  Writing figures only
at the end means staring at a log the whole time and losing everything if the
run is interrupted, so :class:`FigureCheckpoint` plugs into the driver's
``callbacks`` hook and refreshes the same files every ``every`` iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .driver import History
from .mesh import StructuredQuadMesh
from .state import Responses


@dataclass
class FigureCheckpoint:
    """Driver callback that rewrites the result figures as the run proceeds.

    Parameters
    ----------
    outdir:
        Directory for ``design.png``, ``history.png`` and ``checkpoint.npz``.
    mesh:
        The mesh being optimized.
    target_loops:
        Closed target curves, drawn over the modal profile.
    target_curves:
        Open target curves, drawn without a closing segment.
    every:
        Iteration stride.  Figures are always written on the first iteration
        too, so something exists to look at almost immediately.
    method:
        ``"density"`` or ``"levelset"``; picks the figure style and is stored
        in the snapshot.
    """

    outdir: Path
    mesh: StructuredQuadMesh
    target_loops: Sequence[np.ndarray]
    every: int = 25
    dpi: int = 140
    target_curves: Sequence[np.ndarray] = ()
    method: str = "density"
    _written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self, iteration: int, state, responses: Responses, history: History
    ) -> None:
        if iteration != 1 and iteration % self.every:
            return
        self.write(iteration, state, history)

    def write(self, iteration: int, state, history: History) -> None:
        """Write both figures and a reloadable snapshot for one iteration."""
        # Imported lazily: a headless run that never checkpoints should not
        # have to pay for pyplot.
        import matplotlib.pyplot as plt

        from .postprocess import DESIGN_STYLES, figure_design, figure_history

        mode = state.modes[0]
        fig = figure_design(
            self.mesh,
            state.design.xbar,
            mode.phi_full,
            self.target_loops,
            mode.frequency,
            suptitle=f"iteration {iteration}",
            target_curves=self.target_curves,
            **DESIGN_STYLES[self.method],
        )
        fig.savefig(
            self.outdir / "design.png",
            dpi=self.dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = figure_history(history)
        fig.savefig(
            self.outdir / "history.png",
            dpi=self.dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)

        points, offsets = pack_loops(self.target_loops)
        curve_points, curve_offsets = pack_loops(self.target_curves)
        np.savez_compressed(
            self.outdir / "checkpoint.npz",
            iteration=iteration,
            method=self.method,
            x=state.design.x,
            xbar=state.design.xbar,
            modes=mode.phi_full.reshape(-1, 1),
            frequencies=np.array([mode.frequency]),
            objective=float(history.objective[-1]),
            lx=self.mesh.lx,
            ly=self.mesh.ly,
            nx=self.mesh.nx,
            ny=self.mesh.ny,
            loops=points,
            loop_offsets=offsets,
            curves=curve_points,
            curve_offsets=curve_offsets,
        )
        self._written += 1


def pack_loops(loops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten ragged point arrays into one array plus offsets, for ``.npz``.

    Works for open curves as well as loops: it stores points, not topology.
    """
    arrays = [np.asarray(loop, dtype=float) for loop in loops]
    if not arrays:
        return np.zeros((0, 2)), np.zeros(1, dtype=np.int64)
    offsets = np.cumsum([0] + [len(a) for a in arrays]).astype(np.int64)
    return np.vstack(arrays), offsets


def unpack_loops(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Inverse of :func:`pack_loops`."""
    return [points[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
