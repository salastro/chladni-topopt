"""Four-node Reissner-Mindlin plate element.

Kinematics (Eq. (3) of the paper).  Nodal degrees of freedom are
``(w, theta_x, theta_y)`` with the displacement field

    u = -z * theta_y,   v = z * theta_x,   w = w

giving curvatures and transverse shear strains

    kappa = [-theta_y,x,  theta_x,y,  theta_x,x - theta_y,y]
    gamma = [w,x - theta_y,  w,y + theta_x]

Bending is always integrated with 2x2 Gauss.  The transverse shear term needs
care: full 2x2 integration shear-locks (the plate is thin compared with the
0.02 m elements used in the paper), while the textbook cure of one-point
reduced integration leaves two hourglass mechanisms per element.  Five
zero-energy modes instead of three is harmless in a static compliance problem
but not here, where spurious mechanisms can surface as spurious low-frequency
eigenmodes and derail mode tracking.

The default ``shear_scheme`` is therefore ``"mitc4"`` (Dvorkin & Bathe's
assumed natural strain interpolation), which is both locking-free and
rank-sufficient.  ``"reduced"`` reproduces the plain selective reduced
integration implied by Eq. (3), and ``"full"`` is provided so the locking can
be observed directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# 2x2 Gauss rule on [-1, 1]^2.
_G = 1.0 / np.sqrt(3.0)
_GAUSS_2X2 = (
    ((-_G, -_G), 1.0),
    ((_G, -_G), 1.0),
    ((_G, _G), 1.0),
    ((-_G, _G), 1.0),
)
_GAUSS_1X1 = (((0.0, 0.0), 4.0),)

#: MITC4 tying points: (xi, eta) and which covariant component is sampled.
_TYING = {
    "A": ((0.0, -1.0), 0),
    "B": ((1.0, 0.0), 1),
    "C": ((0.0, 1.0), 0),
    "D": ((-1.0, 0.0), 1),
}

ShearScheme = Literal["mitc4", "reduced", "full"]


def shape_functions(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear shape functions and their natural derivatives.

    Returns ``(N, dN)`` where ``N`` is ``(4,)`` and ``dN`` is ``(2, 4)`` holding
    ``dN/dxi`` and ``dN/deta``.
    """
    n = 0.25 * np.array(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ]
    )
    dn = 0.25 * np.array(
        [
            [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
        ]
    )
    return n, dn


@dataclass(frozen=True)
class MindlinQ4:
    """Reissner-Mindlin plate element.

    Parameters
    ----------
    thickness:
        Plate thickness ``h`` in metres.
    poisson:
        Poisson's ratio.  Young's modulus is *not* stored: it factors out of
        the stiffness matrix and is applied by the material interpolation.
    shear_correction:
        ``kappa`` in Eq. (3); 5/6 or ``pi**2 / 12``.
    shear_scheme:
        Treatment of the transverse shear term; see the module docstring.
    """

    thickness: float
    poisson: float = 0.3
    shear_correction: float = 5.0 / 6.0
    shear_scheme: ShearScheme = "mitc4"

    ndof_per_node: int = 3
    dof_names: tuple[str, ...] = ("w", "theta_x", "theta_y")
    transverse_dof: int = 0

    # -- constitutive matrices (with E factored out) --------------------

    def bending_matrix(self) -> np.ndarray:
        """``D / E`` of Eq. (4)."""
        nu = self.poisson
        return (1.0 / (1.0 - nu**2)) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - nu)],
            ]
        )

    def shear_matrix(self) -> np.ndarray:
        """``D_s / E`` of Eq. (5)."""
        return (1.0 / (2.0 * (1.0 + self.poisson))) * np.eye(2)

    # -- strain-displacement operators ----------------------------------

    @staticmethod
    def _b_bending(dn_xy: np.ndarray) -> np.ndarray:
        """``(3, 12)`` curvature operator from cartesian shape derivatives."""
        b = np.zeros((3, 12))
        for i in range(4):
            dx, dy = dn_xy[0, i], dn_xy[1, i]
            b[0, 3 * i + 2] = -dx
            b[1, 3 * i + 1] = dy
            b[2, 3 * i + 1] = dx
            b[2, 3 * i + 2] = -dy
        return b

    @staticmethod
    def _b_shear_cartesian(n: np.ndarray, dn_xy: np.ndarray) -> np.ndarray:
        """``(2, 12)`` shear operator evaluated directly in cartesian axes."""
        b = np.zeros((2, 12))
        for i in range(4):
            dx, dy = dn_xy[0, i], dn_xy[1, i]
            b[0, 3 * i] = dx
            b[0, 3 * i + 2] = -n[i]
            b[1, 3 * i] = dy
            b[1, 3 * i + 1] = n[i]
        return b

    @staticmethod
    def _b_shear_covariant_row(
        coords: np.ndarray, xi: float, eta: float, component: int
    ) -> np.ndarray:
        """One covariant shear row ``gamma_xi`` or ``gamma_eta`` at ``(xi, eta)``.

        With ``gamma_a = w,a - theta_y * x,a + theta_x * y,a`` for ``a`` in
        ``{xi, eta}``.
        """
        n, dn = shape_functions(xi, eta)
        jac = dn @ coords  # rows: [x,xi  y,xi], [x,eta  y,eta]
        dx_da, dy_da = jac[component]
        row = np.zeros(12)
        for i in range(4):
            row[3 * i] = dn[component, i]
            row[3 * i + 1] = n[i] * dy_da
            row[3 * i + 2] = -n[i] * dx_da
        return row

    def _b_shear_mitc4(
        self, coords: np.ndarray, xi: float, eta: float, jac: np.ndarray
    ) -> np.ndarray:
        """``(2, 12)`` MITC4 shear operator in cartesian axes.

        The covariant strains are interpolated from the four mid-edge tying
        points and then pulled back through the Jacobian.
        """
        rows = {
            key: self._b_shear_covariant_row(coords, pt[0], pt[1], comp)
            for key, (pt, comp) in _TYING.items()
        }
        b_cov = np.empty((2, 12))
        b_cov[0] = 0.5 * (1.0 - eta) * rows["A"] + 0.5 * (1.0 + eta) * rows["C"]
        b_cov[1] = 0.5 * (1.0 + xi) * rows["B"] + 0.5 * (1.0 - xi) * rows["D"]
        return np.linalg.solve(jac, b_cov)

    @staticmethod
    def _n_matrix(n: np.ndarray) -> np.ndarray:
        """``(3, 12)`` shape-function matrix ``[N_i * I_3]``."""
        mat = np.zeros((3, 12))
        for i in range(4):
            mat[0, 3 * i] = n[i]
            mat[1, 3 * i + 1] = n[i]
            mat[2, 3 * i + 2] = n[i]
        return mat

    # -- element matrices -----------------------------------------------

    def unit_matrices(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(ke0, me0)`` for one element, at ``E = 1`` and ``rho = 1``."""
        h = self.thickness
        d_b = self.bending_matrix()
        d_s = self.shear_matrix()
        shear_factor = self.shear_correction * h

        ke = np.zeros((12, 12))
        me = np.zeros((12, 12))
        # Rotatory inertia of Eq. (7), with rho factored out.
        inertia = np.diag([h, h**3 / 12.0, h**3 / 12.0])

        for (xi, eta), weight in _GAUSS_2X2:
            n, dn = shape_functions(xi, eta)
            jac = dn @ coords
            det_j = np.linalg.det(jac)
            dn_xy = np.linalg.solve(jac, dn)

            b_b = self._b_bending(dn_xy)
            ke += weight * det_j * (h**3 / 12.0) * (b_b.T @ d_b @ b_b)

            n_mat = self._n_matrix(n)
            me += weight * det_j * (n_mat.T @ inertia @ n_mat)

            if self.shear_scheme == "mitc4":
                b_s = self._b_shear_mitc4(coords, xi, eta, jac)
                ke += weight * det_j * shear_factor * (b_s.T @ d_s @ b_s)
            elif self.shear_scheme == "full":
                b_s = self._b_shear_cartesian(n, dn_xy)
                ke += weight * det_j * shear_factor * (b_s.T @ d_s @ b_s)

        if self.shear_scheme == "reduced":
            for (xi, eta), weight in _GAUSS_1X1:
                n, dn = shape_functions(xi, eta)
                jac = dn @ coords
                det_j = np.linalg.det(jac)
                dn_xy = np.linalg.solve(jac, dn)
                b_s = self._b_shear_cartesian(n, dn_xy)
                ke += weight * det_j * shear_factor * (b_s.T @ d_s @ b_s)

        # Kill the asymmetry that floating-point accumulation leaves behind.
        return 0.5 * (ke + ke.T), 0.5 * (me + me.T)
