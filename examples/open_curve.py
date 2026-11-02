"""Open targets: nodal lines that run to the plate edge instead of closing.

A nodal line cannot stop in the middle of a plate -- it either closes on
itself or ends on the boundary.  So unlike the letter D of Case 1, which is a
closed loop, these targets are *open* curves whose ends run out to the edges
(see ``examples/shapes.u_curve`` and ``letter_s``).

    python examples/open_curve.py --shape u --quick   # 50 x 60 preview
    python examples/open_curve.py --shape u           # 100 x 120
    python examples/open_curve.py --shape s
    python examples/open_curve.py --shape u --plate sideways --mode 10   # the paper's U

The carrier modes are the ones that worked when tried: mode 6 for the U and
mode 13 for the S, both on the upright plate.  Choosing a carrier is trial and
error.  A topology heuristic -- a mode whose nodal line already joins the same
edges as the target -- found the U's mode but not the S's, since the S failed
on mode 3, the obvious low-order choice.  ``examples/sweep_modes.py`` runs the
search: it tries every mode and ranks the results by how much of the target the
final nodal line traces.  The start-up table here scores each mode by the
amplitude already on the target band; it is local, and it picked the wrong
mode for both shapes.  Override the carrier with ``--mode``.

The design is updated with AdamW.  Results go to ``results/open_<shape>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chladni_topopt.checkpoint import FigureCheckpoint  # noqa: E402
from chladni_topopt.config import (  # noqa: E402
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
    build_problem,
)
from chladni_topopt.indicators import AVERAGE  # noqa: E402
from chladni_topopt.io import save_design, save_history  # noqa: E402
from chladni_topopt.metrics import nodal_line_fit  # noqa: E402
from chladni_topopt.postprocess import (  # noqa: E402
    figure_design,
    figure_history,
    plot_selection,
)
from examples import shapes  # noqa: E402
from examples.shapes import letter_s, u_curve  # noqa: E402

#: Plate outlines.  "sideways" swaps length and width, as the paper does for
#: its Case 2 (the letters D, T and U).  The same modes appear at the same
#: frequencies, rotated -- so "mode 10" means a different pattern per plate.
PLATES = {"upright": (2.0, 2.4), "sideways": (2.4, 2.0)}
ELEMENT = 0.02  # element size in metres, as in the paper's 100 x 120 mesh

#: Each shape's curve, centred on the plate, and its default carrier mode (1-based).
SHAPES = {
    "u": (lambda lx, ly: u_curve(cx=lx / 2, width=1.2, bottom=ly - 1.4, plate=(lx, ly)), 6),
    "s": (
        lambda lx, ly: letter_s(cx=lx / 2, cy=ly / 2, radius=0.4, plate=(lx, ly), hook=70.0),
        13,  # found by examples/sweep_modes.py; modes 1-4 all fail
    ),
}

# Letters drawn as legal nodal patterns (see examples/shapes.py): T and P are
# traced as their outlines, and Y carries one extra stroke, because a three-way
# junction is impossible.
# The carrier mode is None until a sweep has found one; until then --mode is
# required.
for _name, _builder in {
    "t": shapes.letter_t_nodal, "y": shapes.letter_y_nodal, "z": shapes.letter_z_nodal,
    "b": shapes.letter_b_nodal, "m": shapes.letter_m_nodal, "l": shapes.letter_l_nodal,
    "w": shapes.letter_w_nodal, "x": shapes.letter_x_nodal, "p": shapes.letter_p_nodal,
}.items():
    SHAPES[_name] = (lambda lx, ly, _b=_builder: _b((lx, ly)), None)

# The Y as drawn, with its impossible three-way junction -- an experiment, not
# a recommended target.
SHAPES["y_plain"] = (lambda lx, ly: shapes.letter_y_plain((lx, ly)), None)


def build_config(args: argparse.Namespace) -> ProblemConfig:
    make_curve, _ = SHAPES[args.shape]
    lx, ly = PLATES[args.plate]
    curve = make_curve(lx, ly)
    config = ProblemConfig(
        targets=[
            TargetConfig(
                loops=(),
                open_curves=curve,
                mode_index=args.mode - 1,
            )
        ],
        mesh=MeshConfig(
            lx=lx, ly=ly, nx=round(lx / ELEMENT), ny=round(ly / ELEMENT), thickness=0.2
        ),
        optimizer=OptimizerConfig(
            method="adamw", max_iterations=args.iterations, lr=args.lr
        ),
        solver=args.solver,
    )
    return config.coarsen(2) if args.quick else config


def check_ends_are_free(problem, curve: np.ndarray) -> None:
    """Refuse a curve whose ends land inside a clamped region.

    The displacement is zero there already, so the nodal line would not need
    to reach the edge at all -- the target would stop meaning "ends on the
    boundary".
    """
    coords = problem.mesh.node_coords()
    fixed_nodes = set(problem.dofs.fixed // problem.mesh.ndof_per_node)
    if np.allclose(curve[0], curve[-1]):
        return  # a closed loop has no ends
    for end in (curve[0], curve[-1]):
        nearest = int(np.argmin(np.linalg.norm(coords - end, axis=1)))
        if nearest in fixed_nodes:
            raise SystemExit(
                f"the curve ends at {np.round(end, 3)}, inside a clamped corner; "
                "change its parameters in SHAPES"
            )


def print_mode_scores(problem, n_show: int = 8) -> None:
    """How close each mode of the uniform plate already is to the target."""
    analysis = problem.analysis
    state = analysis.solve(problem.x0)
    selection = analysis.selections[0]
    chosen = analysis.trackers[0].index
    print("uniform plate: Avg(L)/Avg(L~) per mode -- lower means the mode")
    print("already has little amplitude on the target band")
    # Always list far enough to include the carrier itself.
    n_show = max(n_show, chosen + 2)
    for i in range(min(n_show, state.solution.full.shape[1])):
        phi = state.solution.full[:, i]
        score = AVERAGE.value(selection.shape, phi) / AVERAGE.value(
            selection.surround, phi
        )
        mark = "  <- carrier" if i == chosen else ""
        print(f"  mode {i + 1:2d}  {state.solution.frequencies[i]:7.2f} Hz  {score:5.3f}{mark}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="half-resolution preview run")
    parser.add_argument("--shape", choices=sorted(SHAPES), default="u")
    parser.add_argument(
        "--plate",
        choices=sorted(PLATES),
        default="upright",
        help="sideways swaps length and width, as in the paper's Case 2",
    )
    parser.add_argument(
        "--mode", type=int, default=None, help="carrier mode, 1-based (default per shape)"
    )
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.02, help="AdamW initial step size")
    parser.add_argument("--solver", choices=["auto", "superlu", "pardiso"], default="auto")
    parser.add_argument("--outdir", type=Path, default=None, help="default results/<shape>")
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=25,
        metavar="N",
        help="refresh the figures every N iterations while running (0 to disable)",
    )
    args = parser.parse_args()
    if args.mode is None:
        args.mode = SHAPES[args.shape][1]
    if args.mode is None:
        parser.error(
            f"no carrier mode chosen for --shape {args.shape} yet; pass --mode, or find "
            f"one with: python examples/sweep_modes.py --shape {args.shape}"
        )
    if args.outdir is None:
        args.outdir = Path("results") / f"open_{args.shape}"

    config = build_config(args)
    problem = build_problem(config)
    curves = list(config.targets[0].open_curves)
    for curve in curves:
        check_ends_are_free(problem, curve)
    print(
        f"mesh {config.mesh.nx} x {config.mesh.ny} "
        f"({problem.mesh.n_elem} elements, {problem.dofs.n_free} free DOFs)\n"
        f"target: {args.shape.upper()} ({len(curves)} curve{'s' if len(curves) > 1 else ''})"
        f" on mode {args.mode}\n"
    )
    print_mode_scores(problem)
    # Scoring solved the eigenproblem once; start the tracker afresh so it
    # locks onto the carrier from the run's own first iteration.
    problem.analysis.trackers[0].reference = None

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        problem.driver.callbacks = [
            FigureCheckpoint(
                outdir=outdir,
                mesh=problem.mesh,
                target_loops=(),
                target_curves=curves,
                every=args.checkpoint,
            )
        ]

    result = problem.run()
    save_design(outdir / "design.npz", result, problem.mesh)
    save_history(outdir / "history.json", result.history)

    mode = result.state.modes[0]
    save = dict(dpi=160, bbox_inches="tight")
    fig = figure_design(
        problem.mesh,
        result.state.design.xbar,
        mode.phi_full,
        (),
        mode.frequency,
        target_curves=curves,
    )
    fig.savefig(outdir / "design.png", facecolor=fig.get_facecolor(), **save)
    plt.close(fig)

    fig = figure_history(result.history)
    fig.savefig(outdir / "history.png", facecolor=fig.get_facecolor(), **save)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    plot_selection(problem.mesh, problem.analysis.selections[0], ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "selection.png", facecolor=fig.get_facecolor(), **save)
    plt.close(fig)

    final = result.history.records[-1]
    formulation = config.formulation
    fit = nodal_line_fit(
        problem.mesh, mode.phi_full, open_curves=curves, dofs=problem.dofs
    )
    print(
        f"\nconverged={result.converged} after {result.iterations} iterations\n"
        f"objective          {final.objective:.4f}\n"
        f"Avg(L, phi) ratio  {final.diagnostics['avg_shape_ratio']:.4f}  (limit {formulation.eps})\n"
        f"Var(L~, phi) ratio {final.diagnostics['var_surround_ratio']:.4f}  (limit {formulation.tau})\n"
        f"volume fraction    {final.diagnostics['volume_fraction']:.4f}\n"
        f"target frequency   {final.target_frequencies[0]:.2f} Hz  "
        f"(spectrum position {final.target_indices[0] + 1})\n"
        f"nodal line fit     {fit.summary()}\n"
        f"written to {outdir}/  (replot any time: python examples/plot_result.py {outdir})"
    )


if __name__ == "__main__":
    main()
