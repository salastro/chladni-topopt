# Chladni-pattern topology optimization

[![tests](https://github.com/salastro/chladni-topopt/actions/workflows/tests.yml/badge.svg)](https://github.com/salastro/chladni-topopt/actions/workflows/tests.yml)

This is a Python reimplementation of

> L. Xue, G. Wen, F. Wang, N. L. Pedersen, *Topology optimization design of
> Chladni patterns for vibration mode manipulability*, Acta Mechanica Sinica
> **40**(4), 423445 (2024). [doi:10.1007/s10409-023-23445-x](https://doi.org/10.1007/s10409-023-23445-x)

It covers the paper's single-mode case (Section 4.1). You pick a vibration
mode of a bi-material plate and a target shape, and the optimizer lays out
the two materials so that the mode's nodal lines, its Chladni pattern, trace
that shape.

```
pip install -r requirements.txt

python examples/case1_letter_D.py --quick   # 50 x 60 preview, a few minutes
python examples/case1_letter_D.py           # the paper's 100 x 120 mesh
python examples/open_curve.py --shape u    # an open U hanging from the top edge
python examples/open_curve.py --shape s    # an open S running edge to edge
python examples/sweep_modes.py --shape s   # try every carrier mode, ranked
python examples/levelset_letters.py --shape d --quick   # level-set method (Wang 2003)
python -m pytest tests/                     # verification suite
```

The paper distributes material. The package also has a second approach that
designs boundary geometry instead, using the level-set method of Wang, Wang &
Guo (2003) to cut holes in a single-material plate. See
[Level-set method](#level-set-method-wang-et-al-2003).

The papers themselves are not in this repository. The citations are above and
in the level-set section.

The density examples use AdamW. MMA, the paper's optimizer, is still in the
package (`OptimizerConfig(method="mma")`), but it does worse on this problem.
[Optimizers](#optimizers) has the comparison.

## Results at a glance

Each figure shows the design, the target (dashed) against the actual nodal
line, and the mode shape. Every run writes more figures under `results/`. That
folder isn't tracked, so rerun the examples to regenerate it.

| Density method (the paper's) | Level-set method (Wang 2003) |
|---|---|
| ![D, mode 5](docs/figures/density_D_mode5.png) | ![D, mode 5](docs/figures/levelset_D_mode5.png) |
| Letter D on mode 5, aluminium/epoxy | Letter D on mode 5, holes in aluminium |
| ![U, open, mode 6](docs/figures/density_U_open_mode6.png) | ![T, mode 9](docs/figures/levelset_T_mode9.png) |
| Open U hanging from the top edge, mode 6 | The paper's T stroke on mode 9 |
| ![S, open, mode 13](docs/figures/density_S_open_mode13.png) | ![U, mode 7](docs/figures/levelset_U_mode7.png) |
| Open S edge to edge, mode 13 | The paper's U on mode 7, which fails at full resolution |

## Seeing the result

A run writes `results/case1/`:

| file | what it is |
|---|---|
| `design.png` | material layout, target vs. actual pattern, mode shape + nodal lines |
| `history.png` | objective and constraint indicators, volume fraction, frequency evolution |
| `selection.png` | the controlled areas, i.e. Fig. 2(b) |
| `design.npz` | design fields, the final mode, and the target curves |
| `history.json` | every iteration's objective, constraints, frequencies and MAC |

A paper-resolution run takes about an hour, so you can watch it as it goes.
`--checkpoint N` (default 25) rewrites `design.png`, `history.png` and a
reloadable `checkpoint.npz` every `N` iterations, starting at iteration 1. Open
`design.png` in a viewer that reloads on change and you can watch the nodal
line move onto the target. `--checkpoint 0` turns this off.

To redraw a saved run without recomputing anything:

```
python examples/plot_result.py results/case1           # finished run
python examples/plot_result.py results/case1 --show    # open a window
python examples/plot_result.py results/case1/checkpoint.npz -o /tmp/peek
```

The `.npz` stores the target curves, so a result replots without rebuilding
the problem. For a custom figure, the functions in `chladni_topopt.postprocess`
(`plot_material`, `plot_mode`, `plot_selection`, `plot_indicators`,
`plot_volume`, `plot_frequencies`) each take an `ax` and a `PlotTheme` you can
swap.

## What the method does

A plate's mode shapes depend on where its material is, so choosing the layout
chooses the pattern. The paper marks two areas around the target curve (its
Fig. 2b). The shape area `L` is a band on the curve itself. The surrounding
area `L̃` is a band on each side of it, separated from `L` by a gap. Two
indicators measure the mode's amplitude over an area:

```
Avg(L, φ) = Lᵀ(φ∘φ) / ΣL              Var(L, φ) = Lᵀ((φ∘φ) − μ)² / ΣL
```

Eq. (15) pushes the amplitude down on the curve and up around it, and keeps
the surrounding amplitude even. That is the condition for a nodal line to lie
along the curve:

```
min  C  = Avg(L̃, φ⁰) / Avg(L̃, φ)
s.t. g₁ = Avg(L, φ)  / Avg(L, φ⁰)  ≤ ε      (ε = 0.01)
     g₂ = Var(L̃, φ) / Var(L̃, φ⁰) ≤ τ      (τ = 0.40)
     g₃ = V − V* ≤ 0
```

Every term is divided by its value for the initial design `φ⁰`, so each starts
at 1. A bordered adjoint gives the gradients (see below). AdamW updates the
design by default, or MMA as in the paper (see [Optimizers](#optimizers)).

## Layout

Every stage is a `typing.Protocol` with one to three methods, and
`ChladniAnalysis` and `TopologyOptimizer` only see the protocol. To replace a
component you write one class and pass it in. There's no subclassing, no edit
to the driver, and no registry.

| Module | Protocol | Implementations |
|---|---|---|
| `mesh.py` | none | `StructuredQuadMesh` |
| `elements/` | `ElementFormulation` | `MindlinQ4` (MITC4 / reduced / full shear) |
| `bc.py` | `BoundaryCondition` | `CornerClamps`, `ClampedEdges`, `SimplySupportedEdges` |
| `material.py` | `MaterialInterpolation` | `SIMPTwoMaterial` (Eq. 10), `SolidVoidErsatz` (level set) |
| `filters.py` | `Filter`, `Projection` | `DensityFilter` (Eq. 12), `ExponentialThreshold` (Eq. 11), `IdentityFilter` |
| `eigen.py` | `EigenSolver` | `ShiftInvertEigen` |
| `linalg.py` | `Factorizer` | `PardisoFactorizer` (default when installed), `SuperLUFactorizer` |
| `tracking.py` | `ModeTracker` | `MACTracker` (Eq. 19), `FixedIndexTracker` |
| `indicators.py` | `Indicator` | `Average` (Eq. 13), `Variance` (Eq. 14) |
| `formulations.py` | `Formulation` | `ConstraintFormulation` (Eq. 15), `MultiObjectiveFormulation` (Eq. 21) |
| `sensitivity.py` | `SensitivityAnalysis` | `AdjointEigenvectorSensitivity` (Eqs. 16-18) |
| `optimizers/` | `Optimizer` | `AdamW` (augmented Lagrangian, default), `MMA` (Svanberg 1987), `HamiltonJacobi` (level set) |
| `levelset/` | `Filter`, `Projection`, `Reinitializer`, `Optimizer` | `NodalToElement`, `WangHeaviside` (Wang Eqs. 39-40), `ContourReinitializer`, `LevelSetUpdate` |
| `regularizers.py` | `Formulation` (wrapper) | `RegularizedFormulation` with `PerimeterTV`; `count_holes` for reporting |

`metrics.py` scores how well a result's nodal line matches the target.
`sweep_modes.py` ranks by it, and `open_curve.py` prints it at the end of a run.
`analysis.py` runs the forward problem and `driver.py` runs the loop of Fig. 3.
`config.py` has `build_problem(cfg)`, which wires up the usual components.
`postprocess.py` and `io.py` handle figures and saving.

Target shapes are plain data on purpose. `examples/shapes.py` returns lists of
`(N, 2)` point arrays in plate coordinates, and anything in that form works,
whether it's hand-drawn points, sampled SVG paths or a traced bitmap.

A target can be made of closed loops, open curves, or both. A nodal line can't
stop in the middle of a plate. It either closes on itself or ends on the
boundary, so an open curve should run all the way out to the edges. `u_curve`
hangs from the top edge, with two straight arms that meet the edge at right
angles and join in a semicircle. `letter_s` runs from the left edge to the
right edge, and each end continues along its own tangent to the edge so there
is no cusp. `examples/open_curve.py --shape u|s` uses both. Open curves differ
from loops in two ways. An open curve gets no closing segment. And "which
side of the line" means left or right of the curve as you walk along its
points, not inside or outside.

```python
from chladni_topopt.selection import build_selection_vectors
loops = [np.array([[x0, y0], [x1, y1], ...])]     # closed loop, metres
curves = [np.array([[0.0, y0], ..., [lx, y1]])]   # open, edge to edge
sel = build_selection_vectors(mesh, dofs, loops, band=..., gap=..., surround=...,
                              open_curves=curves)
# or, through the config:
TargetConfig(loops=(), open_curves=curves, mode_index=2)
```

**The U works.** `python examples/open_curve.py --shape u` (100 × 120, AdamW,
mode 6) converges in 242 iterations with every constraint satisfied:
`Avg(L,φ) = 0.0068` (limit 0.01), `Var(L̃,φ) = 0.394` (limit 0.4), objective
1.51, volume fraction 0.45, at 192 Hz. The nodal line runs down both arms and
follows the rounded bottom closely. Three things differ from the target. The
last 0.2 m or so of each arm flares out towards the clamped top corners,
meeting the edge near x = 0.25 and 1.75 instead of 0.5 and 1.5. A second, small
nodal arc sits just under the top edge inside the U. And another line runs
along the bottom edge. The formulation only controls the neighbourhood of the
target, so lines elsewhere are allowed.

**The paper's carrier for the U, mode 10, also works, but not as cleanly.** The
paper's Case 2 puts the U on mode 10 of the plate turned sideways (2.4 × 2.0).
There, mode 10 has a nodal line hanging from the top edge, the same topology as
the U. `open_curve.py --shape u --plate sideways --mode 10` converges in 251
iterations with `Avg(L,φ) = 0.0042`, going from 400.7 Hz to 372.5 Hz (the paper
starts at 394.35 Hz). The full U forms, but an extra diagonal nodal line
crosses its interior and meets the left arm in an X. On the upright plate,
mode 10 is a different pattern with no line hanging from the top. That run
also "converges", with `Avg(L,φ) = 0.0088` at 317 Hz, but the lines only partly
follow the U and wander off to the bottom edge. It isn't a U. Upright mode 6 is
still the cleanest U.

**Which mode should carry the target? Try them.** I haven't found a cheap test
on the plain plate that predicts the right carrier, so
`examples/sweep_modes.py` searches. It runs the same short optimization on
every candidate mode and plate orientation, then ranks the results by how much
of the target the final nodal line traces
(`chladni_topopt.metrics.nodal_line_fit`).

```
python examples/sweep_modes.py --shape s                  # modes 1-16, both plates
python examples/sweep_modes.py --shape u --modes 4-10 --plates upright
```

It writes `results/sweep_<shape>/summary.csv` and a `contact_sheet.png` with
every run's pattern in rank order. It re-scores runs already on disk instead of
re-optimizing them, so an interrupted sweep picks up where it stopped. The
metric looks at the line itself because the optimizer's own `Avg(L,φ)` can't be
trusted here. It has reported "converged and feasible" for patterns that looked
nothing like the target, by making the neighbourhood of the curve quiet instead
of putting a line along it.

What the searches found, and what didn't work:

- **The start-up amplitude score misleads.** For each mode of the uniform
  plate, the example prints how much amplitude it already has on the target
  band. That score picked the wrong carrier for both shapes (mode 2 for the S,
  mode 4 for the U).
- **Matching edge topology is a hint, not a rule.** Looking for a mode whose
  nodal line already joins the same edges as the target found the U's mode 6,
  whose upper nodal line already hangs from the top edge. For the S it pointed
  to mode 3, a single left-to-right line, and mode 3 failed because the ends ran
  into the top-left and bottom-right clamped corners. A clamped corner has zero
  displacement, so it's a cheap place for a nodal line to end.
- **The S needs a high mode.** Sweeping modes 1-16 on both orientations (quick
  mesh, 150 iterations), modes 1-4 traced 0-13% of the S. Four carriers traced
  over 90%: upright 13 (97%), sideways 9 (92%), sideways 11 (91%) and sideways
  13 (91%). A high mode has nodal lines to spare, and enough curvature to bend
  round both hooks and still reach the side edges.

**The S works on upright mode 13**, now the default for `open_curve.py --shape
s`. At 100 × 120 it converges in 222 iterations with every constraint met:
`Avg(L,φ) = 0.0093`, `Var(L̃,φ) = 0.376`, objective 0.87, at 414 Hz. The nodal
line traces 96.6% of the target. It follows both arcs and both long tails
without a break from the left edge to the right edge. There are two flaws.
Another nodal line touches the upper arc on its left side and leaves a small
notch. And the S is one of a family of roughly parallel nodal lines, which is
normal for a 13th mode and outside the controlled zone.

Sideways mode 11 scored almost as high (96.2% at full resolution), but it looks
worse. Neither tail reaches its edge. Each turns off about 0.3 m short and
closes into a cell, and the metric only counts the tails as traced because
other nodal lines pass nearby. So `traced` checks that the target lies near
some nodal line, not that one continuous line follows it. Use it to shortlist
candidates, then look at the pictures.

### Swapping a component

```python
class MyFilter:                                   # satisfies `Filter`
    def forward(self, x):          return ...     # x -> xtilde
    def backward(self, dc_dxtilde): return ...    # chain rule back to x

analysis = ChladniAnalysis(..., filter=MyFilter(), ...)
```

Any other row of the table works the same way. `build_problem` wires up the
defaults for convenience, but you don't have to use it. Build
`ChladniAnalysis` and `TopologyOptimizer` yourself whenever it gets in the way.

## Implementation notes

**Element.** A Reissner-Mindlin Q4 with `(w, θx, θy)` per node. Bending is
integrated 2×2, and the shear term defaults to MITC4 instead of the one-point
reduced integration that Eq. (3) implies. Reduced integration leaves two
hourglass mechanisms per element (`test_element.py` finds 5 zero-energy modes
instead of 3). That's harmless in a static problem, but in an eigenvalue
problem it can show up as spurious modes and throw off mode tracking.
`MindlinQ4(shear_scheme=...)` still offers both.

**Assembly.** Eqs. (8)-(9) factor `E` and `ρ` out of the element matrices, and
the mesh is uniform, so `ke0` and `me0` are built once. Each assembly then
scatter-adds rescaled copies with precomputed indices. Constrained DOFs are
dropped when the triplets are built, not by slicing the assembled matrix.

**Adjoint (Eqs. 16-18).** For a response `C(φ, λ)` with `R₁ = (K − λM)φ` and
`R₂ = (φᵀMφ − 1)/2`, stationarity of `C + uᵀR₁ + vR₂` gives

```
[ K − λM   Mφ ] [u]   [ −∂C/∂φ ]
[   φᵀM     0 ] [v] = [  ∂C/∂λ ]

dC/dx̄ₑ = uᵀ(∂K/∂x̄ₑ − λ ∂M/∂x̄ₑ)φ + (v/2) φᵀ(∂M/∂x̄ₑ)φ
```

`K − λM` is singular on its own, since `φ` spans its null space. The border
row and column are what make the system solvable. The matrix is the same for
every response of a mode, so it is factorized once per iteration and
back-substituted three times. This matches the paper's Eqs. (16)-(17) up to
their factor of two on `u` and `v`. Finite differences in
`tests/test_sensitivity.py` pin down the convention here.

In the variance gradient the `∂μ/∂φ` term cancels exactly, because
`Σᵢ Lᵢ(φᵢ² − μ) = 0` by the definition of `μ`:

```
∂Avg/∂φ = 2(L∘φ)/ΣL        ∂Var/∂φ = 4(L∘φ∘(φ∘φ − μ))/ΣL
```

**Selection vectors.** Only the transverse DOF of each node is selected, and
constrained nodes are left out. Their displacement is always zero, so counting
them would bias both indicators for reasons that have nothing to do with the
design.

**Lengths are in element widths.** The filter radius and the three zone widths
are stored in element widths, as the paper gives them (`rmin = 4`). So
`coarsen()` divides them by the same factor it divides the mesh by. If they
kept the same element count on a coarser mesh, the physical smoothing radius
would quietly quadruple and the problem would change. In practice this is the
difference between `Avg(L,φ)` reaching 0.05 and stalling at 0.6.

## Optimizers

`OptimizerConfig(method="adamw" | "mma")` picks the update rule. Both satisfy
the same `Optimizer` protocol, so the driver doesn't know which one it has.

AdamW is unconstrained, and this problem has three inequality constraints and
a `[0, 1]` box. `optimizers/adamw.py` adds two pieces:

- **Bounds by projection.** Every step is clipped back into the box.
- **Constraints by a Powell-Hestenes-Rockafellar augmented Lagrangian.** Adam
  descends on `f₀ + Σᵢ (max(0, λᵢ + ρgᵢ)² − λᵢ²) / 2ρ`, and every
  `dual_interval` steps the multipliers update as `λᵢ ← max(0, λᵢ + ρgᵢ)`.

Three settings differ from the textbook defaults:

- **The step size decays.** Adam normalises each coordinate's step, so near a
  solution the step doesn't shrink. At a constant rate the iterates keep
  circling the optimum at a radius of about `lr`, and the multipliers circle
  with them. So the rate decays geometrically,
  `lr_t = max(lr_min, lr · lr_decay^t)`. The defaults take it from 0.02 to
  0.002 over about 600 iterations.
- **`lr` does the job of MMA's move limit.** Adam's normalised step is about
  `lr` per variable per iteration.
- **Weight decay is off.** AdamW's decay pulls every variable towards 0, which
  here means towards epoxy. There's no physical reason for that bias.
  `weight_decay` and `decay_target` are there if you want a regulariser, for
  example decay towards 0.5.

MMA is what the paper uses. Its convex subproblem handles the constraints
exactly at every step, instead of through multipliers that adapt slowly.

Here they are head to head on the quick problem (50 × 60, same setup, 150
iterations). Each row is three runs from the same start, one unchanged and two
nudged by ±1e-9. The trajectories amplify round-off, so two runs that differ
only at that level end at visibly different designs, and one run per setting
can't tell settings apart. Ranges are min-max over the three runs
(`results/adamw_trials/seeds.py`, `seeds.log`).

| | `Avg(L,φ)` ratio (≤ 0.01) | `Var(L̃,φ)` ratio (≤ 0.40) | objective | volume |
|---|---|---|---|---|
| MMA, `move = 0.1` | 0.050-0.061 | 0.377-0.393 | 0.84-0.86 | 0.52-0.53 |
| AdamW, `lr = 0.02` (default) | 0.030-0.031 | 0.297-0.339 | 1.13-1.21 | 0.53-0.54 |
| AdamW, `lr = 0.05` | 0.033 | 0.302-0.419 | 1.01-1.08 | 0.54 |

Two differences are bigger than the spread:

- **AdamW against MMA.** AdamW ends with `Avg(L,φ)` about half of MMA's, and its
  objective above 1 where MMA's is below. Both gaps are several times the
  run-to-run spread. Neither optimizer reaches `ε = 0.01` in 150 iterations.
- **`lr = 0.02` against `0.05`.** The larger step recovers some objective
  (1.01-1.08 against 1.13-1.21) for a slightly higher `Avg(L,φ)` (0.033 against
  0.030). The ranges don't overlap, but the differences are small.

One thing isn't. At `lr = 0.05` the variance ratio goes from comfortably
feasible to over its limit (0.419) across runs that differ only by round-off.
An earlier single-run `lr = 0.01` trial (`compare.log`) isn't repeated here,
and its apparent differences from `lr = 0.02` were within this spread. None of
the runs switched modes.

On the quick mesh the `L` band can't be narrower than one element, 0.04 m, and
`linear_profile_shape_ratio` gives 0.038 for that width. AdamW finishes below
that number by letting the surrounding amplitude fall, which is why its
objective ends above 1. "Tuning notes" below explains what the estimate does
and doesn't claim.

## Performance

**A GPU (or JAX) wouldn't make this faster.** Profiling one iteration on the
paper mesh (35 211 free DOFs) shows where the time goes:

| stage | share |
|---|---|
| eigensolve: shift-invert Lanczos, mostly sparse factorization and triangular solves | ~63% |
| adjoint: factorizing the bordered `[K−λM, Mφ; φᵀM, 0]` system | ~12% |
| everything a GPU parallelizes well (assembly, filter, element `einsum`s, indicators) | ~2% |

About 87% is sparse direct factorization and triangular solves. That's
irregular, memory-bound work that GPUs handle badly at this size, and JAX has
no solver for it at all. `jax.scipy.sparse.linalg` only has iterative methods,
and there is no sparse eigensolver. The problem also needs float64, and a
consumer card like the GTX 1650 Ti runs FP64 at 1/32 of its FP32 rate, slower
than the CPU. Going dense doesn't help. The 35k × 35k matrix is 9.9 GB, and a
dense eigensolve is far more work. Autodiff doesn't help either, because
differentiating through a sparse eigensolve still needs the same hand-written
adjoint. A GPU starts to pay off at millions of DOFs with matrix-free
multigrid, on a datacenter card with full-rate FP64, which is a different
problem.

**The CPU is where the speedups are.** `examples/benchmark_solvers.py` times
real AdamW iterations stage by stage. Paper mesh, idle i7-10750H (6 cores, 12
threads), seconds per iteration:

| configuration | eigen | adjoint | per iteration | speedup |
|---|---|---|---|---|
| SuperLU, 12 modes, default threading (the original code) | 3.23 | 1.38 | 4.75 | 1.0× |
| SuperLU, 12 modes, OpenBLAS held to 1 thread | 2.50 | 1.38 | 3.96 | 1.2× |
| SuperLU, 8 modes, warm start | 2.33 | 1.38 | 3.78 | 1.3× |
| PARDISO, 12 modes | 0.95 | 0.32 | 1.40 | 3.4× |
| **PARDISO, 8 modes, warm start** (the default when installed) | **0.86** | **0.35** | **1.34** | **3.5×** |

A 600-iteration run drops from about 47 minutes to about 13. The gain grows
with mesh size. On the 50 × 60 quick mesh, where the factorizations are small,
it's only about 1.2×.

**The backends compute the same thing, but long runs drift apart.** SuperLU
and PARDISO iterates agree to about 1e-13 for the first 28 or so AdamW
iterations (and to 1e-11 Hz in the benchmark above). After that the gap grows
about 20× per iteration, and by iteration 40 the two runs sit at different
designs. The backend isn't the cause. Two SuperLU runs that differ only in
ARPACK's starting vector split apart the same way from the same iteration. The
optimization amplifies round-off, so any setting that changes the arithmetic
(backend, thread count, BLAS build, warm start) ends at a different but equally
valid local optimum. Compare configurations by their final indicators, not by
matching designs.

The three changes, and what each one is worth:

- **PARDISO** (`pip install pypardiso`, which brings Intel MKL) replaces
  SciPy's single-threaded SuperLU with a supernodal, multithreaded,
  symmetry-aware factorization. It makes the eigensolve 2.7× faster and the
  adjoint 4× faster, and the package picks it up automatically when it's
  installed. To choose a backend yourself, use
  `ProblemConfig(solver="superlu" | "pardiso" | "auto")` or `--solver`. The
  backends sit behind the `Factorizer` protocol in `linalg.py`.
- **OpenBLAS is held to one thread during the eigensolve.** Every Lanczos step
  alternates a triangular solve with a small orthogonalization in numpy's
  OpenBLAS, and the two libraries' spinning thread pools steal cores from each
  other. With PARDISO this alone was worth 5×. In isolation, the eigensolve
  took 3.05 s before and 0.53 s after. The limit only applies inside the
  eigensolve, so importing the package never changes a host program's
  threading. It needs `threadpoolctl`. Without it, set `OPENBLAS_NUM_THREADS=1`.
- **8 modes instead of 12, warm-started from the previous iteration**, saves
  only 5-10%. ARPACK builds a Krylov basis of at least about 20 vectors
  whatever `k` is. An earlier measurement on a busy machine suggested 2×, and
  that didn't hold up. `build_problem` keeps at least 4 modes above the highest
  target, so mode tracking can't lose it.

On a system Python that refuses `pip install` (PEP 668), use a venv that keeps
the system NumPy and SciPy:

```
python3 -m venv --system-site-packages .venv
.venv/bin/pip install pypardiso
.venv/bin/python examples/case1_letter_D.py --quick
```

## Tuning notes

**The band width usually decides whether `ε` is reachable, more than the
optimizer does.** `L` is a band of finite width, so even a mode whose nodal
line runs exactly along the target still has amplitude elsewhere in the band.
Model the mode as linear through the line, `φ ≈ A·d`, and fix `A` so the
surrounding amplitude matches the reference design. That gives

```
Avg(L, φ) / Avg(L, φ⁰)  ≈  [ Avg(L̃, φ⁰) ⟨d²⟩_L ] / [ Avg(L, φ⁰) ⟨d²⟩_L̃ ]
```

`selection.linear_profile_shape_ratio()` evaluates it. For the letter D on the
paper's mesh:

| `band` half-width | estimate for `Avg(L,φ)` | vs. `ε = 0.01` |
|---|---|---|
| 0.5 el (10 mm) | 0.0017 | well under |
| **1 el (20 mm)** | **0.0089** | **just under** |
| 2 el (40 mm) | 0.0424 | 4× over |
| 4 el (80 mm) | 0.1423 | 14× over |

The useful part is the quadratic scaling. Doubling the band roughly quadruples
the ratio a design can reach, so check it before starting a long run. My first
attempt used a 2-element band read off Fig. 2(b) and spent 300 iterations stuck
at `Avg(L,φ) ≈ 0.043`. It looked like a stalled optimizer, but the design was
sitting where that band width puts it, with the nodal line already on the
target. So the default is 1 element.

**It's an estimate, not a bound.** A design can beat both of its assumptions.
It assumes the mode is linear across both zones, but a real mode can stay flat
inside the band and turn up sharply outside it. It also holds the surrounding
amplitude at the reference level (objective `C ≤ 1`), while a design that lets
`Avg(L̃,φ)` fall scales the whole profile down. The AdamW runs above do exactly
that and finish below the estimate, even after correcting by `C`. Use it to
sanity-check `ε`, not to declare a configuration impossible.

**Surrounding-zone width.** The paper's Fig. 2(b) shows `L̃` filling the
target's whole interior, and the 8-element default doesn't (it covers 256 of
the 585 interior nodes of the letter D). Widening it doesn't help, though. At
50 × 60 over 150 iterations:

| `surround` | interior coverage | best `Avg(L,φ)` | final | final `Var(L̃,φ)` |
|---|---|---|---|---|
| 8 el (0.16 m) | 256/585 | 0.049 | **0.052** | 0.390 |
| 16 el (0.32 m) | 392/585 | **0.026** | 0.104 | 0.416 |
| 24 el (0.48 m) | 413/585 | 0.056 | 0.115 | 0.472 |

A wider zone reaches a better minimum along the way but ends worse, because
the variance constraint gets harder to hold over a larger area and the run
oscillates. Coverage tops out near 413/585 whatever you do, since the band and
the gap take up part of the interior themselves. The default stays at 8.

## Level-set method (Wang et al. 2003)

Everything above decides how much aluminium each element holds. The level-set
method of

> M. Y. Wang, X. Wang, D. Guo, *A level set method for structural topology
> optimization*, Comput. Methods Appl. Mech. Engrg. **192** (2003) 227-246

designs the boundary instead. A nodal function `Φ` is positive in material and
negative in holes, and its zero contour is the plate's edge. The plate is all
aluminium, with holes where `Φ < 0`.

```
python examples/levelset_letters.py --shape d --quick          # 50 x 60 preview
python examples/levelset_letters.py --shape t                  # 120 x 100, Wang's HJ update
python examples/levelset_letters.py --shape u --update adamw   # AdamW on nodal Φ instead
```

### How it plugs in

The analysis, the adjoint, the mode tracking and the driver didn't change.
`build_levelset_problem` (in `levelset/config.py`) swaps the parts of the
density chain that describe the design:

| Density method | Level-set method | Wang et al. |
|---|---|---|
| `x`: element densities | `x`: nodal `Φ` | Eq. 38 |
| `DensityFilter` | `NodalToElement`: element value = mean of its 4 nodes | Eq. 38 |
| `ExponentialThreshold` | `WangHeaviside`: smoothed `H(Φ)`, gradient `δ(Φ)` | Eqs. 39-40 |
| `SIMPTwoMaterial` | `SolidVoidErsatz`: void = 1e-3 of the stiffness, 1e-5 of the mass | ersatz material |
| `BetaContinuation` | `NoContinuation` | none |
| `AdamW` | `LevelSetUpdate(HamiltonJacobi)` (default) or `LevelSetUpdate(AdamW)` | Steps 3-5, Eqs. 28, 41, 42 |

The existing adjoint then returns `dC/dΦ = Pᵀ (δ(PΦ) · dC/dx̄)`. That's the
discrete form of Wang's Eq. (28) with `μ ≡ 1`. It's non-zero only within a
Heaviside width of the boundary, which is Wang's narrow band.
`tests/test_levelset.py` checks the whole chain against finite differences.

`HamiltonJacobi` (`optimizers/hamilton_jacobi.py`) does one update in three
steps:

1. It folds the objective and constraints into one sensitivity through the
   augmented Lagrangian in `optimizers/lagrangian.py`, the same one AdamW uses.
   Wang has a single volume multiplier `λ₊` (Eq. 32), and the Chladni
   formulation has three constraints.
2. It smooths the speed `V = −G` over neighbouring nodes (the non-local `μ` of
   Eq. 28) and scales it so its peak is 1.
3. It takes the upwind step of Eq. (41) at a CFL-limited `Δt` (Eq. 42), with
   zero-flux edges (Eq. 14).

`LevelSetUpdate` wraps whichever update you choose. After every step it keeps
the clamps solid, and every 5 steps it resets `Φ` to a signed distance.
`ContourReinitializer` does that by measuring the distance to the extracted
zero contour (Eq. 43).

A level set can't open a hole where there isn't one, so every run starts from
a perforated plate (`levelset/initial.py`, Wang §8).

### Results: D, T, U

The targets come from Xue et al. The D is Case 1's outline on the upright
plate. The T and U are Case 2's strokes on the sideways plate, which I read
off their Fig. 8(a) (`examples/shapes.liang_t_stroke` and `liang_u_stroke`).

Each letter runs as its own single-mode problem. I had to find the carriers
again (quick mesh, 200 iterations, modes 4-14), because a perforated aluminium
plate orders its modes differently from the uniform bi-material plate:

| Letter | Paper's mode | Carrier here | Quick-mesh `traced` | What forms |
|---|---|---|---|---|
| D | 5 | 5 | 84% | the full outline, one closed line |
| T | 7 | 9 | 91% | bar and stem; the stem continues to the top edge (a 3-way junction is impossible) |
| U | 10 | 7 | 61% | both arms; the bottom stays open and the arms run on to the bottom edge |

On the paper's carriers, T traced 44% (mode 7 gives vertical nodal lines only)
and U 20% (mode 10 gives one arm). `traced` is measured with the holes masked
out (`nodal_line_fit(..., solid=xbar)`), since nothing collects on a hole.

At full resolution, 600 HJ iterations of
`examples/levelset_letters.py --shape d|t|u` gave the following. None of the
three runs met the convergence test.

| Letter | Mesh | `traced` | `Avg(L,φ)` (limit 0.01) | Solid fraction (cap 0.55) | Frequency | Verdict by eye |
|---|---|---|---|---|---|---|
| D, mode 5 | 100×120 | 59% | 0.017 | 0.49 | 335 Hz | a closed D; the straight side sits about 5 cm inside the target, the curve follows it |
| T, mode 9 | 120×100 | 85% | 0.019 | 0.51 | 602 Hz | a clean T, with bar, stem, and the stem continuing to the top edge |
| U, mode 7 | 120×100 | 32% | 0.089 | **0.65** | 469 Hz | not a U; the arms lean off the target and the volume cap is broken |

The D's 59% undersells it. The line has no breaks, but stretches of the target
cross holes, and the metric masks those out. The U worked on the quick mesh
(61%) and not at full resolution. The Heaviside width and the CFL step are set
in element widths, so halving the element halves their physical size too. The
full run is therefore a different problem from the preview. The density method
avoids this by scaling its widths in `coarsen()`. The U needs its own tuning at
full resolution, or widths fixed in metres.

### Keeping holes off the line, fewer holes, a moving outline

Three later additions are options of `LevelSetConfig` and flags of
`examples/levelset_letters.py`:

- **A solid corridor along the target** (`protect_target`, on by default, off
  with `--no-protect`). A hole across the target breaks the nodal line right
  where it should show, since nothing collects on a hole. So the nodes within
  the target's band and gap, plus a Heaviside width, stay frozen solid like the
  clamps (`levelset/initial.target_corridor`).
- **A hole-perimeter penalty** (`perimeter_weight`, `--perimeter-weight`,
  default 0.2). The number of holes has no gradient, so the objective gets
  `w · P/P0` instead, where `P` is the total boundary length (the smoothed
  total variation of `x̄`). Small holes shrink away and neighbours merge.
  `RegularizedFormulation` in `regularizers.py` wraps any formulation, and the
  history records `perimeter_ratio` and a `holes` count.
- **A moving plate outline** (`live_edge`, `--live-edge`, experimental). `Φ`
  starts negative in a half-element strip round the plate, so the outline is a
  zero contour and moves like any hole boundary. It can only move inward.

| Corridor + perimeter 0.5 (iteration ~650) | Corridor + moving outline, no penalty (iteration ~450) |
|---|---|
| ![D with corridor](docs/figures/levelset_D_corridor.png) | ![D with live edge](docs/figures/levelset_D_live_edge.png) |

What these runs showed:

- **The corridor works.** No hole touches the line, which stays one closed
  loop.
- **It costs shape control.** The holes that bent the line into the D's
  straight side and corners now have to keep their distance, so the loop
  rounds off.
- **It puts the volume cap at war with the line.** The frozen corridor and the
  clamps start the plate near 70% solid, and the 0.55 cap applies to the whole
  plate. The line constraint pushes boundaries inward and the volume cap pushes
  them outward. Both multipliers grow together and the design stalls. It
  happened on the D, and on the U with mode 7 and with mode 10.
- **The fix is still open.** The options are to measure the cap over the free
  region only, raise `vstar`, or drop the cap for solid/void plates. I lean
  towards the first, since it keeps the cap's meaning without retuning it per
  letter.

### Tuning notes (quick D, 200 iterations)

- **Step size matters most.** At `cfl = 0.5` (half an element per update) the
  objective swings by ±50% every iteration and the D never forms (40%).
  `cfl = 0.2` with 4 smoothing passes gives 82-84%. At 0.3 it falls back to 15%.
- **The perforation matters next.** For the upright plate a 4×5 grid of holes
  works. 3×4 gives 46% and 5×6 gives 13%, because too many small holes leave
  too many boundaries free to move. The hole grid follows the plate's
  orientation (5×4 when sideways).
- **The penalty `rho` barely matters.** Anything from 2 to 30 gives 80-84%.
- **AdamW on `Φ` does worse,** 11-30% for the D. Adam normalises each node's
  step, so nodes with tiny sensitivities move as far as the boundary nodes that
  matter, and nothing plays the role of the smoothing that helps the HJ update.
- **Holes bring their own failure.** A large hole near an edge leaves a thin
  flap of plate. The target mode concentrates on the flap, the region around
  the target goes quiet, and `Avg(L, φ)` counts as satisfied without any line
  forming. You can spot it by an objective far above 2. The density method has
  a milder version of this (the surround amplitude collapses), but holes make
  it easy.
- **No spurious void modes appear.** The mass ratio (1e-5) is well below the
  stiffness ratio (1e-3), so void nodes just follow the plate (max |w| in the
  void is about the same as in the solid) and void elements hold no kinetic
  energy. `test_ersatz_voids_carry_no_local_modes` checks this.
- **Runs never stop on the convergence test.** `Avg(L, φ)` hovers at 0.03-0.05
  of its reference against `ε = 0.01`, so every run uses its full iteration
  budget.

## Where this departs from the paper

| | Paper | Here | Why |
|---|---|---|---|
| Shear integration | 1-point reduced (Eq. 3) | MITC4 by default | reduced integration allows hourglass modes, which can pollute an eigen-analysis |
| Optimizer | MMA | AdamW by default; MMA via `method="mma"` | AdamW reaches about half MMA's `Avg(L,φ)` here; see [Optimizers](#optimizers) for how it handles constraints and bounds |
| Volume limit `V*` | never stated | `0.55` | in Fig. 5(a), Case 1 rises from 0.50 and runs flat at the reported 0.5498, which looks like a binding cap, while Appendix B falls to 0.4286, well inside it |
| Zone widths | shown only in Fig. 2(b) | 1 / 2 / 8 elements | the gap and surround are read off the figure; the 1-element band comes from the estimate above, since with the 2 elements the figure suggests, `Avg(L,φ)` can only get to about 0.04 against an `ε` of 0.01 |
| Convergence (Eq. 20) | index `it−(n−5)` | `it−n−5` | as printed, the paper's index refers to future iterations |
| Convergence gate | objective only | objective and feasibility | the objective levels off long before the eigenvector constraints are met; acting on it alone raises β and stops the run at an infeasible design |
| Constraint scaling | `V − V* ≤ 0` | `V/V* − 1 ≤ 0` | same feasible set, on the same scale as the other constraints, which MMA's subproblem wants |
| Colour maps | rainbow | diverging for signed displacement, 2-colour for the material | the nodal line is the zero level, so a neutral midpoint puts the pattern where the eye goes |

The paper doesn't give Poisson's ratio or the shear correction factor.
`ν = 0.3` and `κ = 5/6` reproduce its 5th-mode frequency for the initial
uniform design to within 1% (227.2 Hz computed, 225.01 Hz reported).

## Verification

`python -m pytest tests/`. GitHub Actions runs the suite on Python 3.11 and
3.14, with and without PARDISO.

- `test_element.py` checks the 3 rigid-body modes (and the 5 that reduced
  integration gives), mass conservation, and second-order convergence to the
  analytic simply-supported Mindlin plate. At `h/a = 1/200` this doubles as the
  locking test.
- `test_sensitivity.py` is the important one. It checks the adjoint against
  central differences for the objective and both eigenvector constraints, at
  two projection sharpnesses and for both formulations.
- `test_mma.py` runs Svanberg's five-variable cantilever benchmark
  (`f₀* = 1.3399`) from feasible and infeasible starts, plus the `m ≥ n` branch
  of the subproblem solver and the move-limit and bound guarantees.
- `test_adamw.py` reaches the same cantilever optimum (to 2e-3, since AdamW is
  first order). It also checks that an active constraint gets a positive
  multiplier and an inactive one keeps zero, and covers box projection, the
  decay schedule and decoupled weight decay.
- `test_filters.py` and `test_selection.py` cover filter adjointness and
  normalisation, projection derivatives, zone geometry, MAC invariances and
  tracking through mode swaps.
- `test_metrics.py` runs the nodal-line metric on fields whose nodal lines are
  known exactly. A line on the target is fully traced, a line elsewhere traces
  nothing, a crossing line counts as stray, and clamp outlines don't count as
  nodal lines.
- `test_open_curves.py` checks that an open curve gets no closing segment and
  that its sign is the side of the line. It also checks that `letter_s` runs
  edge to edge, smooth, point-symmetric and clear of the clamps, and that the
  selection zones straddle it and reach the plate edges. The adjoint still
  matches finite differences on an open target, and open curves survive saving
  and loading (older archives without them still load). They're also drawn
  unclosed.
- `test_letters.py` checks that the letters drawn as nodal patterns are legal
  (ends on the boundary, no three-way junctions) and smooth.
- `test_linalg.py` checks that SuperLU and PARDISO agree on the real `K`
  (symmetric positive definite) and on the bordered adjoint matrix (symmetric
  indefinite, zero corner), for one and several right-hand sides. 8
  warm-started modes match 12 cold ones to 1e-8, and the OpenBLAS thread limit
  applies inside its scope and is undone after. The finite-difference adjoint
  checks in `test_sensitivity.py` also run once per installed backend.
- `test_modularity.py` asserts the protocol claim. Every component satisfies
  its protocol, a filter and an optimizer defined outside the package drop
  straight in, and the chain rule stays correct through a swapped filter
  (checked again by finite differences).
- `test_driver.py` covers the convergence window, the feasibility gate, the
  rule that a run may only stop at `β = β_max`, and a short end-to-end
  optimization.
- `test_levelset.py` covers the level-set pieces. It checks the Heaviside and
  its derivative, the transpose of the nodal-to-element map, the
  reinitialiser, and the upwind step moving a circle at the right speed. It
  checks that ersatz voids carry no local modes, that the perimeter equals
  the boundary length and has the right gradient, and that the corridor stays
  solid. It counts holes, checks that the moving outline moves, and checks
  `dC/dΦ` against finite differences with and without the perimeter penalty.

## Not implemented

Case 2 (three patterns at once), the COMSOL cross-check (Fig. 6), and export
of the isocontour geometry. The mode weights `wₛ` already go through the
formulations, and `ChladniAnalysis` takes a list of targets, so Case 2 would be
a new example rather than a rewrite.
