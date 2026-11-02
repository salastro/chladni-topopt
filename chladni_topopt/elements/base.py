"""Element formulation interface.

An element formulation supplies the *unit* stiffness and mass matrices, i.e.
those evaluated at ``E = 1`` and ``rho = 1``.  The paper's Eqs. (8)-(9) factor
the material properties straight out of the element matrices::

    K = sum_e E_e k_e        M = sum_e rho_e m_e

so on a uniform mesh the element matrices are built once and only rescaled as
the design evolves.  Any replacement formulation (Kirchhoff plate, solid
shell, ...) only has to honour this interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ElementFormulation(Protocol):
    """Produces unit element matrices for a single element geometry."""

    #: Degrees of freedom carried by each node.
    ndof_per_node: int

    #: Human-readable names of the nodal degrees of freedom, in order.
    dof_names: tuple[str, ...]

    #: Index of the transverse-displacement DOF within a node's block.
    transverse_dof: int

    def unit_matrices(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(ke0, me0)`` for an element with the given node coordinates.

        Parameters
        ----------
        coords:
            ``(n_nodes_per_elem, 2)`` array of node positions.

        Returns
        -------
        ke0:
            Stiffness matrix evaluated at ``E = 1``.
        me0:
            Mass matrix evaluated at ``rho = 1``.
        """
        ...
