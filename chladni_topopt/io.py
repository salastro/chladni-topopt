"""Saving and reloading designs and optimization histories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .checkpoint import pack_loops, unpack_loops
from .driver import History, OptimizationResult
from .mesh import StructuredQuadMesh


def save_design(
    path: str | Path,
    result: OptimizationResult,
    mesh: StructuredQuadMesh,
    method: str = "density",
) -> Path:
    """Write the design fields, the final mode and the target curves.

    The target loops and open curves travel with the archive so the result can
    be replotted later without rebuilding the problem -- see
    ``examples/plot_result.py``.  ``method`` records how ``x`` is to be read:
    ``"density"`` (element densities) or ``"levelset"`` (nodal ``Phi``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = result.state
    points, offsets = pack_loops(
        [loop for mode in state.modes for loop in mode.selection.loops]
    )
    curve_points, curve_offsets = pack_loops(
        [curve for mode in state.modes for curve in mode.selection.open_curves]
    )
    np.savez_compressed(
        path,
        loops=points,
        loop_offsets=offsets,
        curves=curve_points,
        curve_offsets=curve_offsets,
        method=method,
        x=result.x,
        xtilde=state.design.xtilde,
        xbar=state.design.xbar,
        eigenvalues=state.solution.values,
        frequencies=state.solution.frequencies,
        target_indices=np.array([m.index for m in state.modes]),
        modes=np.column_stack([m.phi_full for m in state.modes]),
        lx=mesh.lx,
        ly=mesh.ly,
        nx=mesh.nx,
        ny=mesh.ny,
        converged=result.converged,
        iterations=result.iterations,
    )
    return path


def save_history(path: str | Path, history: History) -> Path:
    """Write the per-iteration history as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history.to_dict(), indent=1))
    return path


def load_design(path: str | Path) -> dict:
    """Read back a design archive as a plain dictionary.

    ``loops`` and ``curves`` come back as lists of ``(N, 2)`` point arrays
    rather than the flattened form used on disk.  Archives written before open
    curves existed load with ``curves == []``.
    """
    with np.load(Path(path)) as data:
        loaded = {key: data[key] for key in data.files}
    for key, offsets in (("loops", "loop_offsets"), ("curves", "curve_offsets")):
        if key in loaded and offsets in loaded:
            loaded[key] = unpack_loops(loaded[key], loaded[offsets])
        else:
            loaded[key] = []
    return loaded


def load_history(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
