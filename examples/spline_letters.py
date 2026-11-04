"""The paper's letters with a spline outline: the plate is one closed curve, no holes.

The plate starts as the 2 x 2.4 m rectangle inside a void margin, and its
outline is a periodic cubic B-spline whose control points move along fixed
directions (:mod:`chladni_topopt.spline`).  The outline can grow into the margin
or shrink inside it; the clamped corners and a corridor along the target stay
solid.  There is no volume cap, and a perimeter penalty keeps the outline from
getting wiggly.

    python examples/spline_letters.py --shape d --quick       # 65 x 75 preview
    python examples/spline_letters.py --shape d               # 130 x 150
    python examples/spline_letters.py --shape t --n-ctrl 64

D is Case 1's outline on mode 5; T and U are Case 2's strokes on the sideways
plate, on the carriers found for the level-set method (9 and 7).  MMA updates
the offsets by default.  Results go to ``results/spline_<shape>_<optimizer>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chladni_topopt.checkpoint import FigureCheckpoint  # noqa: E402
from chladni_topopt.config import OptimizerConfig, ProblemConfig, TargetConfig  # noqa: E402
from chladni_topopt.domain import PlateInDomain  # noqa: E402
from chladni_topopt.formulations import ConstraintFormulation  # noqa: E402
from chladni_topopt.io import save_design, save_history  # noqa: E402
from chladni_topopt.metrics import nodal_line_fit  # noqa: E402
from chladni_topopt.postprocess import (  # noqa: E402
    DESIGN_STYLES,
    THEME,
    figure_design,
    figure_history,
    plot_selection,
)
from chladni_topopt.spline import SplineConfig, build_spline_problem  # noqa: E402
from examples.shapes import letter_d, liang_t_stroke, liang_u_stroke  # noqa: E402

ELEMENT = 0.02  # element size in metres, as in the paper's 100 x 120 mesh
STYLE = DESIGN_STYLES["levelset"]  # aluminium/void, holes greyed on the mode plots

#: Letter -> (plate (lx, ly), closed loops, open curves, carrier mode, 1-based).
LETTERS = {
    "d": ((2.0, 2.4), letter_d(cx=1.0, cy=1.2, width=1.0, height=1.2), (), 5),
    "t": ((2.4, 2.0), (), liang_t_stroke(), 9),
    "u": ((2.4, 2.0), (), liang_u_stroke(), 7),
}


def build_config(args: argparse.Namespace) -> tuple[ProblemConfig, SplineConfig]:
    plate, loops, curves, mode = LETTERS[args.shape]
    domain = PlateInDomain(plate=plate, margin=args.domain_margin, element=ELEMENT)
    config = ProblemConfig(
        targets=[
            TargetConfig(
                loops=tuple(domain.shift(loops)),
                open_curves=tuple(domain.shift(curves)),
                mode_index=(args.mode or mode) - 1,
            )
        ],
        mesh=domain.mesh(),
        boundary=domain.clamps(),
        formulation=ConstraintFormulation(vstar=None),
        optimizer=OptimizerConfig(
            method=args.optimizer, max_iterations=args.iterations, move=args.move, lr=args.lr
        ),
        solver=args.solver,
    )
    spline = SplineConfig(
        domain=domain,
        n_ctrl=args.n_ctrl,
        perimeter_weight=args.perimeter_weight,
        protect_target=not args.no_protect,
    )
    return (config.coarsen(2) if args.quick else config), spline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=sorted(LETTERS), default="d")
    parser.add_argument("--optimizer", choices=["mma", "adamw"], default="mma")
    parser.add_argument("--quick", action="store_true", help="half-resolution preview run")
    parser.add_argument("--mode", type=int, default=None, help="carrier mode, 1-based")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--n-ctrl", type=int, default=48, help="B-spline control points")
    parser.add_argument(
        "--move", type=float, default=0.02,
        help="MMA move limit, as a fraction of each offset's range",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="AdamW step size")
    parser.add_argument("--perimeter-weight", type=float, default=SplineConfig.perimeter_weight)
    parser.add_argument("--domain-margin", type=float, default=0.3)
    parser.add_argument(
        "--no-protect", action="store_true",
        help="let the outline cross the target (by default its band and gap stay solid)",
    )
    parser.add_argument("--solver", choices=["auto", "superlu", "pardiso"], default="auto")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=25, metavar="N")
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = Path("results") / f"spline_{args.shape}_{args.optimizer}"

    config, spline = build_config(args)
    problem = build_spline_problem(config, spline)
    spline_map = problem.analysis.filter
    target = config.targets[0]
    loops, curves = list(target.loops), list(target.open_curves)
    print(
        f"mesh {config.mesh.nx} x {config.mesh.ny} ({problem.mesh.n_elem} elements), "
        f"plate {spline.domain.plate[0]} x {spline.domain.plate[1]} m in a "
        f"{spline.domain.margin:g} m margin\n"
        f"target: {args.shape.upper()} on mode {target.mode_index + 1}, "
        f"{spline_map.n} control points, offsets {spline_map.t_lo:+.2f} to "
        f"{spline_map.t_hi:+.2f} m, {args.optimizer}, perimeter weight "
        f"{spline.perimeter_weight}\n"
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        problem.driver.callbacks = [
            FigureCheckpoint(
                outdir=outdir, mesh=problem.mesh, target_loops=loops,
                target_curves=curves, every=args.checkpoint, method="levelset",
            )
        ]

    result = problem.run()
    save_design(outdir / "design.npz", result, problem.mesh, method="levelset")
    save_history(outdir / "history.json", result.history)

    mode = result.state.modes[0]
    xbar = result.state.design.xbar
    save = dict(dpi=160, bbox_inches="tight")
    fig = figure_design(
        problem.mesh, xbar, mode.phi_full, loops, mode.frequency,
        target_curves=curves, **STYLE,
    )
    # The control polygon and the starting rectangle, on the material plot.
    ax = fig.axes[0]
    ctrl = spline_map.control_points(result.x)
    ax.plot(*ctrl[list(range(len(ctrl))) + [0]].T, color=THEME.text_secondary, lw=0.7, marker=".", ms=3)
    start = spline_map.curve(problem.x0)
    ax.plot(*start.T, color=THEME.text_secondary, lw=0.6, ls=":")
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
        problem.mesh, mode.phi_full, loops=loops, open_curves=curves,
        dofs=problem.dofs, solid=xbar,
    )
    t = spline_map.offsets(result.x)
    print(
        f"\nconverged={result.converged} after {result.iterations} iterations\n"
        f"objective          {final.objective:.4f}\n"
        f"Avg(L, phi) ratio  {final.diagnostics['avg_shape_ratio']:.4f}  (limit {formulation.eps})\n"
        f"Var(L~, phi) ratio {final.diagnostics['var_surround_ratio']:.4f}  (limit {formulation.tau})\n"
        f"solid fraction     {final.diagnostics['volume_fraction']:.4f} of the domain\n"
        f"target frequency   {final.target_frequencies[0]:.2f} Hz  "
        f"(spectrum position {final.target_indices[0] + 1})\n"
        f"nodal line fit     {fit.summary()}  (void masked)\n"
        f"outline offsets    {t.min():+.3f} to {t.max():+.3f} m"
        + (f"  (perimeter {final.diagnostics['perimeter_ratio']:.3f} of the initial)\n"
           if "perimeter_ratio" in final.diagnostics else "\n")
        + f"written to {outdir}/  (replot any time: python examples/plot_result.py {outdir})"
    )


if __name__ == "__main__":
    main()
