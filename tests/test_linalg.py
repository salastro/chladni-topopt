"""Factorization backends: they must agree with each other on the real matrices.

Both matrix types that occur in the optimizer are exercised: the stiffness
matrix ``K`` (symmetric positive definite, used by the eigensolver) and the
bordered adjoint matrix (symmetric indefinite, with a structural zero on its
diagonal).  PARDISO cases are skipped when ``pypardiso`` is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from chladni_topopt.eigen import ShiftInvertEigen
from chladni_topopt.linalg import (
    Factorization,
    Factorizer,
    SuperLUFactorizer,
    _upper_with_full_diagonal,
    make_factorizer,
)
from chladni_topopt.material import SIMPTwoMaterial
from chladni_topopt.tracking import mac


@pytest.fixture(scope="module")
def matrices(mesh, assembler, dofs):
    """``K``, ``M`` and the bordered adjoint matrix for a non-uniform design."""
    material = SIMPTwoMaterial(e1=68.5e9, e2=1.0e9, rho1=2700.0, rho2=1180.0)
    x = np.random.default_rng(0).uniform(0.3, 0.7, mesh.n_elem)
    k = assembler.stiffness(material.youngs(x))
    m = assembler.mass(material.density(x))
    solution = ShiftInvertEigen(n_modes=6, factorizer=SuperLUFactorizer()).solve(
        k, m, dofs
    )
    lam, phi, _ = solution.mode(4)
    border = sp.csc_matrix((m @ phi).reshape(-1, 1))
    bordered = sp.bmat(
        [[(k - lam * m).tocsc(), border], [border.T, sp.csc_matrix((1, 1))]],
        format="csc",
    )
    return k, m, bordered


def _cases(matrices):
    k, _, bordered = matrices
    return [("stiffness", k, True), ("bordered", bordered, False)]


def test_backends_satisfy_the_protocol(factorizer, matrices):
    assert isinstance(factorizer, Factorizer)
    k, _, _ = matrices
    assert isinstance(factorizer.factorize(k, definite=True), Factorization)


@pytest.mark.parametrize("which", ["stiffness", "bordered"])
def test_solutions_have_small_residuals(factorizer, matrices, which):
    name, a, definite = next(c for c in _cases(matrices) if c[0] == which)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(a.shape[0])
    x = factorizer.factorize(a, definite=definite).solve(b)
    residual = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    # The bordered matrix is badly scaled (stiffness entries against a unit-
    # normalised border), so its residual floor is higher than K's.
    assert residual < (1e-10 if definite else 1e-6), f"{name}: {residual:.1e}"


@pytest.mark.parametrize("which", ["stiffness", "bordered"])
def test_backends_agree_with_superlu(factorizer, matrices, which):
    """Single and multiple right-hand sides, against the SuperLU reference."""
    _, a, definite = next(c for c in _cases(matrices) if c[0] == which)
    rng = np.random.default_rng(2)
    b1 = rng.standard_normal(a.shape[0])
    b3 = rng.standard_normal((a.shape[0], 3))

    ref = SuperLUFactorizer().factorize(a, definite=definite)
    got = factorizer.factorize(a, definite=definite)

    tol = 1e-10 if definite else 1e-6
    for b in (b1, b3):
        x_ref, x_got = ref.solve(b), got.solve(b)
        assert x_got.shape == b.shape
        assert np.linalg.norm(x_got - x_ref) / np.linalg.norm(x_ref) < tol


def test_upper_triangle_keeps_the_zero_corner(matrices):
    """PARDISO's symmetric types need every diagonal entry stored, zeros included."""
    _, _, bordered = matrices
    upper = _upper_with_full_diagonal(bordered)
    n = bordered.shape[0]
    coo = upper.tocoo()
    assert np.all(coo.row <= coo.col)
    diagonal_stored = np.zeros(n, dtype=bool)
    diagonal_stored[coo.row[coo.row == coo.col]] = True
    assert diagonal_stored.all()
    assert upper[n - 1, n - 1] == 0.0
    # And nothing was lost: upper + strict upper transposed reconstructs it.
    rebuilt = upper + sp.triu(upper, k=1).T
    assert abs(rebuilt - bordered).max() < 1e-12 * abs(bordered).max()


def test_unknown_solver_name_is_rejected():
    with pytest.raises(ValueError, match="unknown solver"):
        make_factorizer("cholmod")


def test_eigensolver_backends_and_warm_start_agree(factorizer, matrices, dofs):
    """8 warm-started modes on any backend match 12 cold SuperLU modes.

    The warm start only changes how quickly ARPACK converges, so its answer
    must match a cold start's to eigensolver precision.  The target mode's
    vectors are compared by MAC, which ignores sign.
    """
    k, m, _ = matrices
    reference = ShiftInvertEigen(
        n_modes=12, factorizer=SuperLUFactorizer(), warm_start=False
    ).solve(k, m, dofs)

    solver = ShiftInvertEigen(n_modes=8, factorizer=factorizer, warm_start=True)
    solver.solve(k, m, dofs)  # prime the warm start
    warm = solver.solve(k, m, dofs)

    rel = np.abs(warm.values - reference.values[:8]) / reference.values[:8]
    assert rel.max() < 1e-8
    assert mac(reference.reduced[:, 4], warm.reduced[:, 4:5])[0] > 1 - 1e-6
    # M-normalisation is load-bearing for the adjoint.
    norms = np.einsum("ik,ik->k", warm.reduced, m @ warm.reduced)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_openblas_is_limited_inside_the_scope_and_restored_after():
    """The thread limit must actually bite, and must not leak out.

    A first version keyed the limit by the wrong name and was silently
    ignored -- the eigensolve stayed ~5x slower with no error anywhere -- so
    the effect is checked, not assumed.
    """
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from chladni_topopt.linalg import openblas_single_threaded

    def openblas_threads():
        return [
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["internal_api"] == "openblas"
        ]

    before = openblas_threads()
    if not before:
        pytest.skip("numpy is not linked against OpenBLAS here")
    with openblas_single_threaded():
        assert all(n == 1 for n in openblas_threads())
    assert openblas_threads() == before
