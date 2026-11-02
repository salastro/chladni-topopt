"""Nodal level-set values -> one value per element.

Wang et al. store ``Phi`` on the finite-element nodes and interpolate it with
the element shape functions, Eq. (38).  The material of an element is taken
from ``Phi`` at its centre, which for a bilinear quad is the mean of its four
corner values.  That map is linear, so it sits in the
:class:`~chladni_topopt.filters.Filter` slot of the analysis chain and its
``backward`` is the exact transpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..mesh import StructuredQuadMesh


@dataclass
class NodalToElement:
    """``Phi_e = mean of Phi at the element's four nodes``."""

    mesh: StructuredQuadMesh
    matrix: sp.csr_matrix = field(init=False, repr=False)

    def __post_init__(self) -> None:
        conn = self.mesh.connectivity()
        rows = np.repeat(np.arange(self.mesh.n_elem), 4)
        self.matrix = sp.csr_matrix(
            (np.full(conn.size, 0.25), (rows, conn.ravel())),
            shape=(self.mesh.n_elem, self.mesh.n_node),
        )

    def forward(self, phi: np.ndarray) -> np.ndarray:
        return self.matrix @ phi

    def backward(self, dc_dphi_elem: np.ndarray) -> np.ndarray:
        return self.matrix.T @ dc_dphi_elem
