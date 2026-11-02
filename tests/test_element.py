"""The Reissner-Mindlin element: rank, consistency and accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.assembly import Assembler
from chladni_topopt.bc import DofSet, SimplySupportedEdges
from chladni_topopt.eigen import ShiftInvertEigen
from chladni_topopt.elements.mindlin_q4 import MindlinQ4
from chladni_topopt.mesh import StructuredQuadMesh

COORDS = np.array([[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]])


@pytest.mark.parametrize("scheme,expected_nullity", [("mitc4", 3), ("full", 3), ("reduced", 5)])
def test_zero_energy_modes(scheme, expected_nullity):
    """MITC4 is rank-sufficient; one-point shear integration is not.

    Three zero-energy modes are the rigid-body motions of a plate element.  The
    two extra ones under ``reduced`` are the hourglass mechanisms that motivate
    using MITC4 as the default.
    """
    ke, _ = MindlinQ4(thickness=0.2, shear_scheme=scheme).unit_matrices(COORDS)
    eigenvalues = np.linalg.eigvalsh(ke)
    nullity = int(np.sum(eigenvalues < 1e-9 * eigenvalues.max()))
    assert nullity == expected_nullity


def test_rigid_body_modes_are_stress_free():
    """Translation and the two rotations store no strain energy."""
    ke, _ = MindlinQ4(thickness=0.2).unit_matrices(COORDS)
    scale = np.abs(ke).max()
    modes = {
        "translation": lambda x, y: (1.0, 0.0, 0.0),
        "rotation about x": lambda x, y: (y, -1.0, 0.0),
        "rotation about y": lambda x, y: (x, 0.0, 1.0),
    }
    for name, field in modes.items():
        v = np.concatenate([field(*c) for c in COORDS])
        assert abs(v @ ke @ v) < 1e-12 * scale, name


def test_mass_matrix_is_positive_definite_and_conserves_mass():
    """The translational block must integrate to ``rho * h * A``."""
    element = MindlinQ4(thickness=0.2)
    _, me = element.unit_matrices(COORDS)
    assert np.linalg.eigvalsh(me).min() > 0.0
    area = 0.02 * 0.02
    assert me[0::3, 0::3].sum() == pytest.approx(element.thickness * area, rel=1e-12)


def test_matrices_are_symmetric():
    ke, me = MindlinQ4(thickness=0.2).unit_matrices(COORDS)
    assert np.allclose(ke, ke.T, atol=1e-18)
    assert np.allclose(me, me.T, atol=1e-18)


def _simply_supported_frequencies(nel: int, h: float, e: float, nu: float, rho: float):
    mesh = StructuredQuadMesh(1.0, 1.0, nel, nel)
    dofs = DofSet.from_mesh(mesh, SimplySupportedEdges())
    asm = Assembler(mesh, MindlinQ4(thickness=h, poisson=nu), dofs)
    k = asm.stiffness(np.full(mesh.n_elem, e))
    m = asm.mass(np.full(mesh.n_elem, rho))
    return ShiftInvertEigen(n_modes=4).solve(k, m, dofs).frequencies


def test_thin_plate_matches_analytic_solution():
    """A thin simply-supported square plate must approach the Kirchhoff limit.

    ``f_mn = (pi/2)(m^2/a^2 + n^2/a^2) sqrt(D / (rho h))``.  This is also the
    locking test: at ``h/a = 1/200`` a shear-locking element overpredicts these
    frequencies by orders of magnitude.
    """
    h, e, nu, rho, a = 0.005, 200e9, 0.3, 8000.0, 1.0
    d = e * h**3 / (12.0 * (1.0 - nu**2))
    exact = np.array(
        sorted(
            (np.pi / 2) * ((m / a) ** 2 + (n / a) ** 2) * np.sqrt(d / (rho * h))
            for m in range(1, 3)
            for n in range(1, 3)
        )
    )

    computed = _simply_supported_frequencies(40, h, e, nu, rho)
    assert np.all(np.abs(computed - exact) / exact < 0.01)


def test_convergence_is_second_order():
    """Halving the element size must cut the frequency error by roughly four."""
    h, e, nu, rho = 0.005, 200e9, 0.3, 8000.0
    d = e * h**3 / (12.0 * (1.0 - nu**2))
    exact = (np.pi / 2) * 2.0 * np.sqrt(d / (rho * h))  # the (1,1) mode

    errors = [
        abs(_simply_supported_frequencies(nel, h, e, nu, rho)[0] - exact) / exact
        for nel in (10, 20, 40)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] / errors[2] > 3.0
