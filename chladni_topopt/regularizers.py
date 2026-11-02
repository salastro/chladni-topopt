"""Design regularisers that ride on top of any formulation.

The number of holes in a design is what one would like to keep small, but it
is topological -- it changes in jumps and has no gradient.  The standard
differentiable stand-in is the total boundary length: every hole costs its
circumference, so penalising perimeter shrinks small holes until they vanish
and pulls neighbouring ones together.  It works on the physical density, so
it applies to a level-set design (where ``xbar = H(Phi)``) and to a density
design alike, and needs nothing from the adjoint: its derivative goes in the
``dxbar`` block of :class:`~chladni_topopt.state.Responses`, which the
sensitivity step already chains back to the design variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

from .formulations import Formulation
from .mesh import StructuredQuadMesh
from .state import DesignState, Responses, TargetMode


@dataclass
class PerimeterTV:
    """Boundary length as the smoothed total variation of ``xbar``.

    ::

        P = sum over interior element edges  l * (sqrt((xbar_a - xbar_b)^2 + eps^2) - eps)

    For a sharp 0/1 field this is the length of the material/void boundary;
    a smoothed Heaviside spreads the jump over a few elements without changing
    the sum.  Only edges between two elements count, so a hole that opens onto
    the plate's outer edge is not charged for the plate edge itself.  ``eps``
    keeps the square root differentiable where neighbours are equal.
    """

    mesh: StructuredQuadMesh
    eps: float = 1e-3
    _diff: sp.csr_matrix = field(init=False, repr=False)
    _length: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nx, ny = self.mesh.nx, self.mesh.ny
        index = np.arange(nx * ny).reshape(nx, ny)  # element (i, j) is i * ny + j
        pairs, lengths = [], []
        # Neighbours along x share an edge of length dy, and along y one of dx.
        for a, b, length in (
            (index[:-1, :], index[1:, :], self.mesh.dy),
            (index[:, :-1], index[:, 1:], self.mesh.dx),
        ):
            pairs.append(np.column_stack([a.ravel(), b.ravel()]))
            lengths.append(np.full(a.size, length))
        pairs = np.vstack(pairs)
        n_edge = len(pairs)
        rows = np.repeat(np.arange(n_edge), 2)
        vals = np.tile([1.0, -1.0], n_edge)
        self._diff = sp.csr_matrix(
            (vals, (rows, pairs.ravel())), shape=(n_edge, self.mesh.n_elem)
        )
        self._length = np.concatenate(lengths)

    def __call__(self, xbar: np.ndarray) -> tuple[float, np.ndarray]:
        """``(P, dP/dxbar)``."""
        jump = self._diff @ xbar
        root = np.sqrt(jump**2 + self.eps**2)
        value = float(self._length @ (root - self.eps))
        grad = self._diff.T @ (self._length * jump / root)
        return value, grad


def count_holes(mesh: StructuredQuadMesh, xbar: np.ndarray, threshold: float = 0.5) -> int:
    """Connected void regions (``xbar < threshold``, edge-adjacent elements).

    Notches that open onto the plate edge count as holes too.  For reporting
    only: the count is what the perimeter penalty stands in for.
    """
    void = np.asarray(xbar).reshape(mesh.nx, mesh.ny) < threshold
    _, n = ndimage.label(void)
    return int(n)


@dataclass
class RegularizedFormulation:
    """Any formulation, with ``weight * P / P0`` added to its objective.

    ``P0`` is the perimeter of the first design evaluated, so the penalty
    starts at ``weight`` whatever the mesh or initial perforation -- the same
    normalisation the Chladni indicators use.  Constraints pass through
    unchanged.
    """

    inner: Formulation
    perimeter: PerimeterTV
    weight: float = 0.2
    reference: float | None = field(default=None, init=False)

    @property
    def constraint_names(self) -> tuple[str, ...]:
        return self.inner.constraint_names

    def responses(self, modes: list[TargetMode], design: DesignState) -> Responses:
        base = self.inner.responses(modes, design)
        value, grad = self.perimeter(design.xbar)
        if self.reference is None:
            if value <= 0.0:
                raise ValueError(
                    "the initial design has no boundary, so its perimeter cannot "
                    "normalise the penalty; start from a perforated design"
                )
            self.reference = value

        values = base.values.copy()
        dxbar = base.dxbar.copy()
        values[0] += self.weight * value / self.reference
        dxbar[0] += self.weight / self.reference * grad
        mesh = self.perimeter.mesh
        diagnostics = dict(
            base.diagnostics,
            perimeter_ratio=value / self.reference,
            holes=float(count_holes(mesh, design.xbar)),
        )
        return Responses(
            names=base.names,
            values=values,
            dphi=base.dphi,
            dlam=base.dlam,
            dxbar=dxbar,
            diagnostics=diagnostics,
        )
