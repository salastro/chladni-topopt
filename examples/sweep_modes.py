"""Try a target on many carrier modes and rank them by the nodal line itself.

Picking the carrier mode is trial and error, so this runs the same short
optimization on every candidate mode (and plate orientation) and scores each
result with :func:`chladni_topopt.metrics.nodal_line_fit` -- the fraction of
the target the final nodal line actually traces -- rather than with the
optimizer's own indicators, which can look fine while the pattern is wrong.

    python examples/sweep_modes.py --shape s                  # modes 1-16, both plates
    python examples/sweep_modes.py --shape u --modes 4-10 --plates upright
    python examples/sweep_modes.py --shape s --modes 5,9 --full --iterations 600

Runs use exactly the configuration of ``examples/open_curve.py`` (AdamW), on
the 50 x 60 quick mesh unless ``--full``.  Runs already saved in ``--outdir``
are re-scored rather than re-optimized (``--rerun`` forces them), so an
interrupted sweep resumes where it stopped.  Writes, under ``--outdir``: one
folder per run (``design.npz``, ``history.json``), ``summary.csv``, and
``contact_sheet.png`` with every run's pattern in rank order.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_modes(text: str) -> list[int]:
    """``"1-16"``, ``"3,6,10"`` or a mix like ``"2-4,9"`` -> sorted mode numbers."""
    modes: set[int] = set()
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            modes.update(range(int(lo), int(hi) + 1))
        else:
            modes.add(int(part))
    return sorted(modes)


def _config(shape: str, plate: str, mode: int, iterations: int, quick: bool):
    sys.path.insert(0, str(ROOT))
    from examples.open_curve import build_config

    return build_config(
        argparse.Namespace(
            shape=shape, plate=plate, mode=mode, iterations=iterations,
            quick=quick, lr=0.02, solver="auto",
        )
    )


def score_run(shape, plate, mode, iterations, quick, outdir) -> dict:
    """Score a finished run from what it saved, so sweeps can be resumed."""
    from chladni_topopt.bc import DofSet
    from chladni_topopt.io import load_design, load_history
    from chladni_topopt.metrics import nodal_line_fit

    config = _config(shape, plate, mode, iterations, quick)
    mesh = config.mesh.build()
    dofs = DofSet.from_mesh(mesh, config.boundary)
    run_dir = Path(outdir) / f"{plate}_mode{mode}"
    data = load_design(run_dir / "design.npz")
    final = load_history(run_dir / "history.json")["records"][-1]
    fit = nodal_line_fit(
        mesh, data["modes"][:, 0],
        loops=list(config.targets[0].loops),
        open_curves=list(config.targets[0].open_curves),
        dofs=dofs,
    )
    return dict(
        traced=fit.traced,
        stray=fit.stray,
        mean_gap_mm=1000 * fit.mean_gap,
        objective=final["objective"],
        avg_shape=final["diagnostics"]["avg_shape_ratio"],
        var_surround=final["diagnostics"]["var_surround_ratio"],
        freq_hz=final["target_frequencies"][0],
        final_position=final["target_indices"][0] + 1,
        converged=bool(data["converged"]),
        iterations=int(data["iterations"]),
        error="",
    )


def run_one(job: tuple) -> dict:
    """One optimization, in its own process; returns the scores.

    A run whose ``design.npz`` already exists is only re-scored, unless
    ``rerun`` is set -- so an interrupted sweep picks up where it stopped.
    """
    shape, plate, mode, iterations, quick, outdir, rerun = job
    sys.path.insert(0, str(ROOT))
    tic = time.perf_counter()
    row = {"shape": shape, "plate": plate, "mode": mode, "reused": False}
    run_dir = Path(outdir) / f"{plate}_mode{mode}"
    try:
        if rerun or not (run_dir / "design.npz").exists():
            from chladni_topopt.config import build_problem
            from chladni_topopt.io import save_design, save_history

            problem = build_problem(_config(shape, plate, mode, iterations, quick), verbose=False)
            problem.driver.verbose = False
            result = problem.run()
            save_design(run_dir / "design.npz", result, problem.mesh)
            save_history(run_dir / "history.json", result.history)
        else:
            row["reused"] = True
        row.update(score_run(shape, plate, mode, iterations, quick, outdir))
    except Exception as exc:  # one bad mode must not sink the sweep
        row.update(error=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(time.perf_counter() - tic, 1)
    return row


def rank_key(row: dict):
    if row.get("error"):
        return (1, 0.0, 0.0)
    return (0, -row["traced"], row["stray"])


def contact_sheet(rows: list[dict], outdir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT))
    from chladni_topopt.io import load_design
    from chladni_topopt.mesh import StructuredQuadMesh
    from chladni_topopt.postprocess import plot_mode

    good = [r for r in rows if not r.get("error")]
    cols = 8
    nrows = max(1, -(-len(good) // cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(2.4 * cols, 2.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for rank, (row, ax) in enumerate(zip(good, axes.flat), start=1):
        data = load_design(outdir / f"{row['plate']}_mode{row['mode']}" / "design.npz")
        mesh = StructuredQuadMesh(
            float(data["lx"]), float(data["ly"]), int(data["nx"]), int(data["ny"])
        )
        ax.set_axis_on()
        plot_mode(
            mesh, data["modes"][:, 0], ax=ax,
            target_loops=list(data["loops"]), target_curves=list(data["curves"]),
            title=(
                f"#{rank} {row['plate'][:4]} mode {row['mode']}\n"
                f"traced {100 * row['traced']:.0f}%  stray {row['stray']:.2f}"
            ),
        )
        ax.title.set_fontsize(7.5)
    fig.tight_layout()
    path = outdir / "contact_sheet.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="s")
    parser.add_argument("--plates", nargs="+", default=["upright", "sideways"])
    parser.add_argument("--modes", default="1-16", help='e.g. "1-16" or "3,6,10"')
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--full", action="store_true", help="100 x 120 instead of the quick mesh")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--rerun", action="store_true", help="re-optimize runs already saved in --outdir"
    )
    args = parser.parse_args()

    outdir = args.outdir or ROOT / "results" / f"sweep_{args.shape}"
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (args.shape, plate, mode, args.iterations, not args.full, str(outdir), args.rerun)
        for plate in args.plates
        for mode in parse_modes(args.modes)
    ]

    # Share the cores between workers instead of letting every PARDISO call
    # try to use all of them.  Spawned workers inherit this environment.
    os.environ["MKL_NUM_THREADS"] = str(max(1, (os.cpu_count() or 4) // args.jobs))
    print(
        f"{len(jobs)} runs of {args.iterations} AdamW iterations "
        f"({'100 x 120' if args.full else 'quick mesh'}), {args.jobs} at a time",
        flush=True,
    )

    rows = []
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for row in pool.imap_unordered(run_one, jobs):
            rows.append(row)
            took = "saved run" if row.get("reused") else f"{row['seconds']:.0f}s"
            status = row["error"] or (
                f"traced {100 * row['traced']:5.1f}%  stray {row['stray']:.2f}  "
                f"obj {row['objective']:.2f}"
            )
            print(
                f"  [{len(rows):2d}/{len(jobs)}] {row['plate']:8s} mode {row['mode']:2d}  "
                f"{status}  ({took})",
                flush=True,
            )

    rows.sort(key=rank_key)
    fields = [
        "shape", "plate", "mode", "traced", "stray", "mean_gap_mm", "objective",
        "avg_shape", "var_surround", "freq_hz", "final_position", "converged",
        "iterations", "seconds", "error",
    ]
    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'rank':>4} {'plate':8s} {'mode':>4} {'traced':>7} {'stray':>6} "
          f"{'gap mm':>7} {'obj':>6} {'Avg(L)':>7} {'Hz':>7}")
    for rank, row in enumerate(rows, start=1):
        if row.get("error"):
            print(f"{rank:4d} {row['plate']:8s} {row['mode']:4d}  ERROR {row['error']}")
            continue
        print(
            f"{rank:4d} {row['plate']:8s} {row['mode']:4d} {100 * row['traced']:6.1f}% "
            f"{row['stray']:6.2f} {row['mean_gap_mm']:7.1f} {row['objective']:6.2f} "
            f"{row['avg_shape']:7.4f} {row['freq_hz']:7.1f}"
        )
    sheet = contact_sheet(rows, outdir)
    print(f"\nwrote {outdir / 'summary.csv'} and {sheet}")


if __name__ == "__main__":
    main()
