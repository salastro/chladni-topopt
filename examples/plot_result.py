"""Render figures from a saved result, without re-running the optimization.

Works on a finished run (``design.npz`` + ``history.json``) or on the
``checkpoint.npz`` a run in progress is still refreshing.

    python examples/plot_result.py results/case1
    python examples/plot_result.py results/case1 --show
    python examples/plot_result.py results/case1/checkpoint.npz -o /tmp/peek
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

import matplotlib.pyplot as plt  # noqa: E402

from chladni_topopt.driver import History, IterationRecord  # noqa: E402
from chladni_topopt.io import load_design, load_history  # noqa: E402
from chladni_topopt.mesh import StructuredQuadMesh  # noqa: E402
from chladni_topopt.postprocess import (  # noqa: E402
    DESIGN_STYLES,
    figure_design,
    figure_history,
)


def history_from_json(data: dict) -> History:
    """Rebuild a :class:`History` from the saved JSON."""
    import numpy as np

    history = History()
    for row in data["records"]:
        history.append(
            IterationRecord(
                iteration=row["iteration"],
                objective=row["objective"],
                constraints=np.array(row["constraints"]),
                frequencies=np.array(row["frequencies"]),
                target_indices=tuple(row["target_indices"]),
                target_frequencies=tuple(row["target_frequencies"]),
                macs=tuple(row["macs"]),
                switched=row["switched"],
                beta=row["beta"] if row["beta"] is not None else float("nan"),
                error=row["error"] if row["error"] is not None else float("inf"),
                diagnostics=row["diagnostics"],
                elapsed=row["elapsed"],
            )
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="a results directory, or a .npz file directly",
    )
    parser.add_argument("-o", "--outdir", type=Path, default=None)
    parser.add_argument("--show", action="store_true", help="open a window instead of only writing files")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    if args.path.is_dir():
        npz = args.path / "design.npz"
        if not npz.exists():
            npz = args.path / "checkpoint.npz"
        history_path = args.path / "history.json"
    else:
        npz = args.path
        history_path = npz.parent / "history.json"

    if not npz.exists():
        raise SystemExit(
            f"no design.npz or checkpoint.npz under {args.path}. "
            "A run writes design.npz when it finishes, and checkpoint.npz "
            "every few iterations while it is going."
        )

    outdir = args.outdir or npz.parent
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_design(npz)
    mesh = StructuredQuadMesh(
        float(data["lx"]), float(data["ly"]), int(data["nx"]), int(data["ny"])
    )
    phi = data["modes"][:, 0]
    frequency = float(data["frequencies"][0 if "iteration" in data else int(data["target_indices"][0])])
    label = f"iteration {int(data['iteration'])}" if "iteration" in data else None

    written = []
    fig = figure_design(
        mesh,
        data["xbar"],
        phi,
        list(data["loops"]),
        frequency,
        suptitle=label,
        target_curves=list(data["curves"]),
        **DESIGN_STYLES[str(data.get("method", "density"))],
    )
    path = outdir / "design.png"
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    written.append(path)
    if not args.show:
        plt.close(fig)

    if history_path.exists():
        history = history_from_json(load_history(history_path))
        fig = figure_history(history)
        path = outdir / "history.png"
        fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        written.append(path)
        if not args.show:
            plt.close(fig)

    print("wrote:")
    for path in written:
        print(f"  {path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
