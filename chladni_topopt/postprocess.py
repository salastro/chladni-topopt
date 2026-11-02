"""Figures: material distribution, modal profiles with nodal lines, histories.

Colour choices follow the job each encoding does rather than the paper's
rainbow.  Modal displacement is a *signed* quantity whose zero is the whole
point of the plot, so it gets a diverging map with a neutral midpoint -- the
nodal line then falls out as the pale band, and the explicit zero contour sits
on top of it.  The two-material design field gets a blend between one colour
per material.  Convergence histories are split into separate panels instead of
sharing a twin axis, because two y-scales on one frame invite misreading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .driver import History
from .mesh import StructuredQuadMesh
from .selection import SelectionVectors


@dataclass(frozen=True)
class PlotTheme:
    """Colours and weights, kept in one place so a whole figure set can be reskinned."""

    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    grid: str = "#dedcd6"
    muted: str = "#b8b6ae"
    #: Categorical slots 1-3, validated for CVD separation on the light surface.
    series: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
    #: Two-material endpoints: soft (epoxy) and strong (aluminium).
    soft: str = "#1baf7a"
    strong: str = "#eda100"
    #: Diverging poles for signed modal displacement.
    negative: str = "#2a78d6"
    positive: str = "#e34948"
    nodal: str = "#0b0b0b"
    target: str = "#7a1fa2"
    #: Holes of a level-set design, drawn over the modal profile.
    void: str = "#d9d7d0"

    @property
    def material_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "material", [self.soft, "#f2f0e9", self.strong]
        )

    @property
    def mode_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "mode", [self.negative, "#f4f2ec", self.positive]
        )


THEME = PlotTheme()

#: ``figure_design`` keywords for each design parameterisation, keyed by the
#: ``method`` a result is saved with.
DESIGN_STYLES = {
    "density": dict(material_labels=("aluminium", "epoxy"), mask_voids=False),
    "levelset": dict(material_labels=("aluminium", "void"), mask_voids=True),
}


def _extent(mesh: StructuredQuadMesh) -> tuple[float, float, float, float]:
    return (0.0, mesh.lx, 0.0, mesh.ly)


def _label_headroom(ax, n_iter: int, fraction: float = 0.22) -> None:
    """Widen the x-range so end-of-line direct labels fit inside the frame."""
    ax.set_xlim(left=0.5, right=max(n_iter * (1.0 + fraction), n_iter + 1))


def _style_axes(ax, theme: PlotTheme = THEME) -> None:
    ax.set_facecolor(theme.surface)
    for spine in ax.spines.values():
        spine.set_color(theme.grid)
    ax.tick_params(colors=theme.text_secondary, labelsize=8)
    ax.xaxis.label.set_color(theme.text_secondary)
    ax.yaxis.label.set_color(theme.text_secondary)
    ax.title.set_color(theme.text_primary)


def element_field(mesh: StructuredQuadMesh, values: np.ndarray) -> np.ndarray:
    """Reshape a per-element array to an image indexed ``[y, x]``."""
    return values.reshape(mesh.nx, mesh.ny).T


def node_field(
    mesh: StructuredQuadMesh, phi_full: np.ndarray, component: int = 0
) -> np.ndarray:
    """Extract one nodal component of a global vector as an image ``[y, x]``."""
    return phi_full[mesh.node_dofs(component)].reshape(mesh.nx + 1, mesh.ny + 1).T


def plot_material(
    mesh: StructuredQuadMesh,
    xbar: np.ndarray,
    ax=None,
    theme: PlotTheme = THEME,
    title: str = "Material distribution",
    labels: tuple[str, str] = ("aluminium", "epoxy"),
):
    """Design field: strong material against soft, or ``labels`` for other
    pairs such as a level-set design's material and void."""
    ax = ax or plt.gca()
    im = ax.imshow(
        element_field(mesh, xbar),
        origin="lower",
        extent=_extent(mesh),
        cmap=theme.material_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax, theme)
    ax.legend(
        handles=[
            Line2D([], [], marker="s", ls="", ms=8, color=theme.strong, label=labels[0]),
            Line2D([], [], marker="s", ls="", ms=8, color=theme.soft, label=labels[1]),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncols=2,
        frameon=False,
        fontsize=8,
        labelcolor=theme.text_secondary,
    )
    return im


def plot_mode(
    mesh: StructuredQuadMesh,
    phi_full: np.ndarray,
    ax=None,
    theme: PlotTheme = THEME,
    target_loops: Sequence[np.ndarray] | None = None,
    title: str | None = None,
    target_curves: Sequence[np.ndarray] | None = None,
    solid: np.ndarray | None = None,
):
    """Modal profile with its nodal lines, optionally over the target curve.

    ``target_loops`` (closed) and ``target_curves`` (open, drawn without a
    closing segment) are plain sequences of ``(N, 2)`` point arrays, so a
    saved result can be replotted without rebuilding the selection vectors.
    ``solid`` (element densities) greys out holes and hides the nodal lines
    inside them, where nothing could collect.
    """
    ax = ax or plt.gca()
    w = node_field(mesh, phi_full)
    peak = np.abs(w).max()
    w = w / peak if peak > 0 else w

    xs = np.linspace(0.0, mesh.lx, mesh.nx + 1)
    ys = np.linspace(0.0, mesh.ly, mesh.ny + 1)

    im = ax.imshow(
        w,
        origin="lower",
        extent=_extent(mesh),
        cmap=theme.mode_cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    if solid is not None:
        holes = element_field(mesh, solid) < 0.5
        # A node is in a hole when every element around it is.
        padded = np.pad(holes, 1, mode="edge")
        node_hole = (
            padded[:-1, :-1] & padded[1:, :-1] & padded[:-1, 1:] & padded[1:, 1:]
        )
        w = np.ma.masked_where(node_hole, w)
        ax.imshow(
            np.ma.masked_where(~holes, np.ones(holes.shape)),
            origin="lower",
            extent=_extent(mesh),
            cmap=LinearSegmentedColormap.from_list("void", [theme.void, theme.void]),
            interpolation="nearest",
        )
    ax.contour(xs, ys, w, levels=[0.0], colors=[theme.nodal], linewidths=1.4)

    for loop in target_loops or ():
        closed = np.vstack([loop, loop[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=theme.target, lw=1.4, ls=(0, (4, 2)))
    for curve in target_curves or ():
        curve = np.asarray(curve)
        ax.plot(curve[:, 0], curve[:, 1], color=theme.target, lw=1.4, ls=(0, (4, 2)))
    # Open curves end on the plate edge, where a line would otherwise be
    # drawn outside the image and change the axis limits.
    ax.set_xlim(0.0, mesh.lx)
    ax.set_ylim(0.0, mesh.ly)

    ax.set_title(title or "Modal profile", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax, theme)
    return im


def plot_selection(
    mesh: StructuredQuadMesh,
    selection: SelectionVectors,
    ax=None,
    theme: PlotTheme = THEME,
):
    """Reproduce Fig. 2(b): the shape area, the gap and the surrounding area."""
    ax = ax or plt.gca()
    nd = mesh.node_dofs(0)
    zones = np.zeros(mesh.n_node)
    zones[selection.surround[nd] > 0] = 1.0
    zones[selection.shape[nd] > 0] = 2.0
    img = zones.reshape(mesh.nx + 1, mesh.ny + 1).T

    cmap = LinearSegmentedColormap.from_list(
        "zones", [theme.grid, theme.series[2], theme.target], N=3
    )
    im = ax.imshow(
        img, origin="lower", extent=_extent(mesh), cmap=cmap, vmin=0, vmax=2,
        interpolation="nearest",
    )
    ax.set_title("Controlled areas", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax, theme)
    ax.legend(
        handles=[
            Line2D([], [], marker="s", ls="", ms=8, color=theme.target, label="shape area $L$"),
            Line2D([], [], marker="s", ls="", ms=8, color=theme.series[2], label=r"surrounding $\tilde{L}$"),
            Line2D([], [], marker="s", ls="", ms=8, color=theme.grid, label="unmarked / gap"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncols=3,
        frameon=False,
        fontsize=7.5,
        labelcolor=theme.text_secondary,
    )
    return im


def plot_indicators(history: History, ax=None, theme: PlotTheme = THEME):
    """Objective and the two eigenvector indicators, on a log scale.

    Direct labels as well as the legend: one of the series sits below 3:1
    contrast on this surface, so identity may not rest on colour alone.
    """
    ax = ax or plt.gca()
    it = np.arange(1, len(history) + 1)
    series = [
        ("objective", history.objective, theme.series[0]),
        (r"Avg$(L,\phi)$ ratio", history.diagnostic("avg_shape_ratio"), theme.series[1]),
        (r"Var$(\tilde{L},\phi)$ ratio", history.diagnostic("var_surround_ratio"), theme.series[2]),
    ]
    for label, values, color in series:
        ax.plot(it, np.maximum(values, 1e-6), color=color, lw=2.0, label=label)
        ax.annotate(
            label,
            xy=(it[-1], max(values[-1], 1e-6)),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.5,
            va="center",
        )

    ax.set_yscale("log")
    _label_headroom(ax, len(history))
    ax.set_xlabel("iteration")
    ax.set_ylabel("normalised value")
    ax.set_title("Objective and constraint indicators", fontsize=10)
    ax.grid(True, which="both", color=theme.grid, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style_axes(ax, theme)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=theme.text_secondary, loc="best")
    return ax


def plot_volume(history: History, ax=None, theme: PlotTheme = THEME):
    """Volume fraction, on its own frame rather than a twin axis."""
    ax = ax or plt.gca()
    it = np.arange(1, len(history) + 1)
    vol = history.diagnostic("volume_fraction")
    ax.plot(it, vol, color=theme.series[1], lw=2.0)
    ax.annotate(
        f"{vol[-1]:.4f}",
        xy=(it[-1], vol[-1]),
        xytext=(4, 0),
        textcoords="offset points",
        color=theme.series[1],
        fontsize=8,
        va="center",
    )
    _label_headroom(ax, len(history))
    ax.set_xlabel("iteration")
    ax.set_ylabel("volume fraction")
    ax.set_title("Volume fraction", fontsize=10)
    ax.grid(True, color=theme.grid, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style_axes(ax, theme)
    return ax


def plot_frequencies(history: History, ax=None, theme: PlotTheme = THEME):
    """Spectrum evolution, with the tracked target mode picked out."""
    ax = ax or plt.gca()
    it = np.arange(1, len(history) + 1)
    freqs = history.frequencies
    for column in range(freqs.shape[1]):
        ax.plot(it, freqs[:, column], color=theme.muted, lw=0.8, alpha=0.8)

    targets = history.target_frequencies
    for s in range(targets.shape[1]):
        ax.plot(it, targets[:, s], color=theme.series[0], lw=2.0, label=f"target mode {s + 1}")
        ax.annotate(
            f"{targets[-1, s]:.1f} Hz",
            xy=(it[-1], targets[-1, s]),
            xytext=(4, 0),
            textcoords="offset points",
            color=theme.series[0],
            fontsize=8,
            va="center",
        )

    _label_headroom(ax, len(history))
    ax.set_xlabel("iteration")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Frequency evolution", fontsize=10)
    ax.grid(True, color=theme.grid, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _style_axes(ax, theme)
    ax.legend(
        handles=[
            Line2D([], [], color=theme.series[0], lw=2.0, label="tracked target mode"),
            Line2D([], [], color=theme.muted, lw=0.8, label="other modes"),
        ],
        frameon=False,
        fontsize=7.5,
        labelcolor=theme.text_secondary,
        loc="best",
    )
    return ax


def figure_design(
    mesh: StructuredQuadMesh,
    xbar: np.ndarray,
    phi_full: np.ndarray,
    target_loops: Sequence[np.ndarray],
    frequency: float,
    theme: PlotTheme = THEME,
    suptitle: str | None = None,
    target_curves: Sequence[np.ndarray] = (),
    material_labels: tuple[str, str] = ("aluminium", "epoxy"),
    mask_voids: bool = False,
):
    """The paper's Fig. 4: material layout, target comparison, resulting CP.

    ``target_loops`` are closed target curves and ``target_curves`` open ones;
    either may be empty.  For a solid/void design pass ``mask_voids=True`` so
    the holes are drawn as holes on the modal plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    fig.patch.set_facecolor(theme.surface)
    solid = xbar if mask_voids else None

    plot_material(mesh, xbar, ax=axes[0], theme=theme, labels=material_labels)
    plot_mode(
        mesh,
        phi_full,
        ax=axes[1],
        theme=theme,
        target_loops=target_loops,
        target_curves=target_curves,
        title="Target (dashed) vs. actual CP",
        solid=solid,
    )
    im = plot_mode(
        mesh,
        phi_full,
        ax=axes[2],
        theme=theme,
        title=f"Mode shape and CP  -  {frequency:.2f} Hz",
        solid=solid,
    )

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("normalised displacement", color=theme.text_secondary, fontsize=8)
    cbar.ax.tick_params(colors=theme.text_secondary, labelsize=8)
    cbar.outline.set_edgecolor(theme.grid)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color=theme.text_primary)
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    else:
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    return fig


def figure_history(history: History, theme: PlotTheme = THEME):
    """The paper's Fig. 5, as three separate frames instead of twinned axes."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.patch.set_facecolor(theme.surface)
    plot_indicators(history, ax=axes[0], theme=theme)
    plot_volume(history, ax=axes[1], theme=theme)
    plot_frequencies(history, ax=axes[2], theme=theme)
    fig.tight_layout()
    return fig
