"""The paper's letters D, T and U by the level-set method of Wang et al. (2003).

Same plate, supports, targets and Chladni formulation as the density examples,
but the design is the plate's own boundary: the zero contour of a level-set
function ``Phi`` that cuts holes into a single-material (aluminium) plate.
The run starts from a perforated plate, because a level set can move and
merge holes but not open new ones.

    python examples/levelset_letters.py --shape d --quick           # 50 x 60 preview
    python examples/levelset_letters.py --shape t                   # 120 x 100
    python examples/levelset_letters.py --shape u --update adamw    # AdamW on Phi

Each letter is shaped on its own mode, as a single-mode problem, with the
paper's targets: D is Case 1's closed outline on the upright 2 x 2.4 m plate;
T (a bar and stem) and U (an open stroke) are Case 2's, on the sideways
2.4 x 2.0 m plate.  The carriers are not the paper's.  Its modes 5, 7 and 10
belong to the uniform two-material plate; a perforated aluminium plate orders
its modes differently, and a carrier sweep found:

* D -- mode 5, as in the paper.
* T -- mode 9.  The paper's mode 7 gives vertical nodal lines only.
* U -- mode 7.  The paper's mode 10 traces one arm.

``--update hj`` (default) is Wang et al.'s upwind Hamilton-Jacobi step;
``--update adamw`` updates the nodal values of ``Phi`` with AdamW instead.
Results go to ``results/levelset_<shape>_<update>/``.
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
from chladni_topopt.config import (  # noqa: E402
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
)
from chladni_topopt.io import save_design, save_history  # noqa: E402
from chladni_topopt.levelset import LevelSetConfig, build_levelset_problem  # noqa: E402
from chladni_topopt.metrics import nodal_line_fit  # noqa: E402
from chladni_topopt.regularizers import count_holes  # noqa: E402
from chladni_topopt.postprocess import (  # noqa: E402
    DESIGN_STYLES,
    figure_design,
    figure_history,
    plot_selection,
)
from examples.shapes import letter_d, liang_t_stroke, liang_u_stroke  # noqa: E402

ELEMENT = 0.02  # element size in metres, as in the paper's 100 x 120 mesh

#: Letter -> (plate (lx, ly), closed loops, open curves, carrier mode (1-based),
#: initial holes along x and y).  The hole grid follows the plate's orientation.
LETTERS = {
    "d": ((2.0, 2.4), letter_d(cx=1.0, cy=1.2, width=1.0, height=1.2), (), 5, (4, 5)),
    "t": ((2.4, 2.0), (), liang_t_stroke(), 9, (5, 4)),
    "u": ((2.4, 2.0), (), liang_u_stroke(), 7, (5, 4)),
}


def build_config(args: argparse.Namespace) -> tuple[ProblemConfig, LevelSetConfig]:
    (lx, ly), loops, curves, mode, holes = LETTERS[args.shape]
    config = ProblemConfig(
        targets=[
            TargetConfig(
                loops=loops,
                open_curves=curves,
                mode_index=(args.mode or mode) - 1,
            )
        ],
        mesh=MeshConfig(
            lx=lx, ly=ly, nx=round(lx / ELEMENT), ny=round(ly / ELEMENT), thickness=0.2
        ),
        optimizer=OptimizerConfig(max_iterations=args.iterations),
        solver=args.solver,
    )
    levelset = LevelSetConfig(
        update=args.update,
        holes=tuple(args.holes or holes),
        protect_target=not args.no_protect,
        perimeter_weight=args.perimeter_weight,
        live_edge=args.live_edge,
    )
    return (config.coarsen(2) if args.quick else config), levelset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=sorted(LETTERS), default="d")
    parser.add_argument("--update", choices=["hj", "adamw"], default="hj")
    parser.add_argument("--quick", action="store_true", help="half-resolution preview run")
    parser.add_argument("--mode", type=int, default=None, help="carrier mode, 1-based (default per letter)")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument(
        "--holes", type=int, nargs=2, default=None, metavar=("NX", "NY"),
        help="initial perforation, holes along x and y (default per letter)",
    )
    parser.add_argument(
        "--perimeter-weight", type=float, default=LevelSetConfig.perimeter_weight,
        help="weight of the hole-perimeter penalty P/P0 on the objective (0 to disable)",
    )
    parser.add_argument(
        "--live-edge", action="store_true",
        help="let the plate outline move too, not only the holes (experimental)",
    )
    parser.add_argument(
        "--no-protect", action="store_true",
        help="let holes cross the target (by default its band and gap stay solid)",
    )
    parser.add_argument("--solver", choices=["auto", "superlu", "pardiso"], default="auto")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=25,
        metavar="N",
        help="refresh the figures every N iterations while running (0 to disable)",
    )
    args = parser.parse_args()
    if args.outdir is None:
        args.outdir = Path("results") / f"levelset_{args.shape}_{args.update}"

    config, levelset = build_config(args)
    problem = build_levelset_problem(config, levelset)
    target = config.targets[0]
    loops, curves = list(target.loops), list(target.open_curves)
    print(
        f"mesh {config.mesh.nx} x {config.mesh.ny} "
        f"({problem.mesh.n_elem} elements, {problem.dofs.n_free} free DOFs)\n"
        f"target: {args.shape.upper()} on mode {target.mode_index + 1}, "
        f"level-set update {args.update}, {levelset.holes[0]} x {levelset.holes[1]} "
        f"initial holes\n"
        f"target corridor {'solid' if levelset.protect_target else 'unprotected'}, "
        f"perimeter weight {levelset.perimeter_weight}, "
        f"outline {'moving' if levelset.live_edge else 'fixed'}\n"
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        problem.driver.callbacks = [
            FigureCheckpoint(
                outdir=outdir,
                mesh=problem.mesh,
                target_loops=loops,
                target_curves=curves,
                every=args.checkpoint,
                method="levelset",
            )
        ]

    result = problem.run()
    save_design(outdir / "design.npz", result, problem.mesh, method="levelset")
    save_history(outdir / "history.json", result.history)

    mode = result.state.modes[0]
    xbar = result.state.design.xbar
    save = dict(dpi=160, bbox_inches="tight")
    fig = figure_design(
        problem.mesh,
        xbar,
        mode.phi_full,
        loops,
        mode.frequency,
        target_curves=curves,
        **DESIGN_STYLES["levelset"],
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
        problem.mesh, mode.phi_full, loops=loops, open_curves=curves,
        dofs=problem.dofs, solid=xbar,
    )
    print(
        f"\nconverged={result.converged} after {result.iterations} iterations\n"
        f"objective          {final.objective:.4f}\n"
        f"Avg(L, phi) ratio  {final.diagnostics['avg_shape_ratio']:.4f}  (limit {formulation.eps})\n"
        f"Var(L~, phi) ratio {final.diagnostics['var_surround_ratio']:.4f}  (limit {formulation.tau})\n"
        f"solid fraction     {final.diagnostics['volume_fraction']:.4f}  (limit {formulation.vstar})\n"
        f"target frequency   {final.target_frequencies[0]:.2f} Hz  "
        f"(spectrum position {final.target_indices[0] + 1})\n"
        f"nodal line fit     {fit.summary()}  (holes masked)\n"
        f"holes              {count_holes(problem.mesh, xbar)}"
        + (
            f"  (perimeter {final.diagnostics['perimeter_ratio']:.3f} of the initial)\n"
            if "perimeter_ratio" in final.diagnostics
            else "\n"
        )
        + f"written to {outdir}/  (replot any time: python examples/plot_result.py {outdir})"
    )


if __name__ == "__main__":
    main()
