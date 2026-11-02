"""Case 1 of the paper (Section 4.1): the letter D as the CP of a single mode.

A 2 m x 2.4 m x 0.2 m aluminium/epoxy plate, clamped over squares at its four
corners, is designed so that the nodal lines of its 5th mode trace a D.

    python examples/case1_letter_D.py            # paper mesh, 100 x 120
    python examples/case1_letter_D.py --quick    # 50 x 60, same physical setup
    python examples/case1_letter_D.py --multi-objective   # Appendix B instead

The design is updated with AdamW.  The paper's MMA is still available in the
package (``OptimizerConfig(method="mma")``), but AdamW gives better results
here -- see the "Optimizers" section of the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chladni_topopt.config import (
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
    build_problem,
)
from chladni_topopt.formulations import ConstraintFormulation, MultiObjectiveFormulation
from chladni_topopt.checkpoint import FigureCheckpoint
from chladni_topopt.io import save_design, save_history
from chladni_topopt.postprocess import figure_design, figure_history, plot_selection
from examples.shapes import letter_d


def build_config(args: argparse.Namespace) -> ProblemConfig:
    mesh = MeshConfig(lx=2.0, ly=2.4, nx=100, ny=120, thickness=0.2)
    formulation = (
        MultiObjectiveFormulation(alpha=0.9801, beta=0.0099, gamma=0.01)
        if args.multi_objective
        else ConstraintFormulation(eps=0.01, tau=0.4)
    )
    config = ProblemConfig(
        targets=[
            TargetConfig(
                loops=letter_d(cx=1.0, cy=1.2, width=1.0, height=1.2),
                mode_index=4,  # the 5th mode, 0-based
                band_elements=1.0,
                gap_elements=2.0,
                surround_elements=8.0,
            )
        ],
        mesh=mesh,
        formulation=formulation,
        optimizer=OptimizerConfig(
            method=args.optimizer,
            max_iterations=args.iterations,
            move=args.move,
            lr=args.lr,
        ),
        solver=args.solver,
    )
    return config.coarsen(2) if args.quick else config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="half-resolution preview run")
    parser.add_argument("--multi-objective", action="store_true", help="use Eq. (21) instead of Eq. (15)")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--optimizer", choices=["adamw", "mma"], default="adamw")
    parser.add_argument("--lr", type=float, default=0.02, help="AdamW initial step size")
    parser.add_argument("--move", type=float, default=0.1, help="MMA move limit")
    parser.add_argument("--outdir", type=Path, default=Path("results/case1"))
    parser.add_argument(
        "--solver",
        choices=["auto", "superlu", "pardiso"],
        default="auto",
        help="sparse factorization backend; auto uses PARDISO when pypardiso is installed",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=25,
        metavar="N",
        help="refresh the figures every N iterations while running (0 to disable)",
    )
    args = parser.parse_args()

    config = build_config(args)
    problem = build_problem(config)
    print(
        f"mesh {config.mesh.nx} x {config.mesh.ny} "
        f"({problem.mesh.n_elem} elements, {problem.dofs.n_free} free DOFs)\n"
        f"target: letter D on mode {config.targets[0].mode_index + 1}\n"
    )

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    loops = list(config.targets[0].loops)
    if args.checkpoint:
        problem.driver.callbacks = [
            FigureCheckpoint(
                outdir=outdir, mesh=problem.mesh, target_loops=loops,
                every=args.checkpoint,
            )
        ]
        print(f"refreshing {outdir}/design.png every {args.checkpoint} iterations\n")

    result = problem.run()
    save_design(outdir / "design.npz", result, problem.mesh)
    save_history(outdir / "history.json", result.history)

    mode = result.state.modes[0]
    selection = problem.analysis.selections[0]

    fig = figure_design(
        problem.mesh, result.state.design.xbar, mode.phi_full, loops, mode.frequency
    )
    fig.savefig(outdir / "design.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig = figure_history(result.history)
    fig.savefig(outdir / "history.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.8))
    plot_selection(problem.mesh, selection, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "selection.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    final = result.history.records[-1]
    print(
        f"\nconverged={result.converged} after {result.iterations} iterations\n"
        f"objective          {final.objective:.4f}\n"
        f"Avg(L, phi) ratio  {final.diagnostics['avg_shape_ratio']:.4f}  (limit {getattr(config.formulation, 'eps', float('nan'))})\n"
        f"Var(L~, phi) ratio {final.diagnostics['var_surround_ratio']:.4f}  (limit {getattr(config.formulation, 'tau', float('nan'))})\n"
        f"volume fraction    {final.diagnostics['volume_fraction']:.4f}\n"
        f"target frequency   {final.target_frequencies[0]:.2f} Hz  (mode index {final.target_indices[0]})\n"
        f"written to {outdir}/  (replot any time: python examples/plot_result.py {outdir})"
    )


if __name__ == "__main__":
    main()
