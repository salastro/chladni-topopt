"""Sparse direct factorizations, behind a swappable interface.

Almost all of an iteration's time goes into two factorizations: ``K`` for the
shift-invert eigensolve and the bordered adjoint matrix for the sensitivities.
This module lets the backend change without touching either caller.

Two backends ship:

* :class:`SuperLUFactorizer` -- SciPy's ``splu``.  Always available,
  single-threaded, general LU.  The fallback.
* :class:`PardisoFactorizer` -- Intel MKL PARDISO through ``pypardiso``
  (``pip install pypardiso``).  Supernodal and multithreaded, and it exploits
  symmetry, which ``splu`` does not.

Callers say whether a matrix is ``definite``.  ``K`` is symmetric positive
definite; the bordered adjoint matrix ``[[K - lam M, M phi], [phi^T M, 0]]``
is symmetric *indefinite* and must never be factorized as if it were SPD.
On this problem PARDISO's SPD path happens to fail loudly on it (error -4,
zero pivot), but nothing guarantees that in general.  The flag is per call
precisely so one setting can never cover both.
"""

from __future__ import annotations

import contextlib
import importlib.util
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


@runtime_checkable
class Factorization(Protocol):
    """A factorized matrix that can solve for one or several right-hand sides."""

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve ``A x = b`` for ``b`` of shape ``(n,)`` or ``(n, k)``."""
        ...


@runtime_checkable
class Factorizer(Protocol):
    """Produces factorizations of symmetric sparse matrices."""

    #: Short backend name, for logs and benchmark tables.
    name: str

    def factorize(self, a: sp.spmatrix, *, definite: bool) -> Factorization:
        """Factorize a symmetric matrix.

        ``definite=True`` promises the matrix is positive definite; pass
        ``False`` for anything else, including every bordered/saddle-point
        system.
        """
        ...


# -- SuperLU ------------------------------------------------------------------


@dataclass
class _SuperLUFactorization:
    lu: object

    def solve(self, b: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b, dtype=float))


@dataclass(frozen=True)
class SuperLUFactorizer:
    """SciPy's SuperLU.  Ignores ``definite``: it is a general LU either way."""

    name: str = "superlu"

    def factorize(self, a: sp.spmatrix, *, definite: bool) -> Factorization:
        return _SuperLUFactorization(splu(sp.csc_matrix(a)))


# -- PARDISO ------------------------------------------------------------------

#: PARDISO matrix types.
_REAL_SPD = 2
_REAL_SYMMETRIC_INDEFINITE = -2


def _upper_with_full_diagonal(a: sp.spmatrix) -> sp.csr_matrix:
    """Upper triangle in CSR with every diagonal entry *stored*.

    PARDISO's symmetric matrix types read only the upper triangle and require
    each diagonal position to be present in the sparsity pattern, even when
    its value is zero.  The bordered adjoint matrix has exactly such a zero,
    in the corner, which SciPy does not store.  Adding a stored zero diagonal
    in COO form keeps the entries: COO -> CSR conversion sums duplicates but
    does not prune explicit zeros.
    """
    upper = sp.triu(sp.coo_matrix(a), format="coo")
    n = a.shape[0]
    rows = np.concatenate([upper.row, np.arange(n)])
    cols = np.concatenate([upper.col, np.arange(n)])
    data = np.concatenate([upper.data, np.zeros(n)])
    out = sp.coo_matrix((data, (rows, cols)), shape=a.shape).tocsr()
    out.sort_indices()
    return out


class _PardisoFactorization:
    """One PARDISO handle holding one factorization.

    The MKL memory behind a handle is outside Python's control, so it is
    released explicitly when the object goes away.  Without that, a run
    would leak two factorizations' worth of memory every iteration.
    """

    def __init__(self, a_upper: sp.csr_matrix, mtype: int) -> None:
        from pypardiso import PyPardisoSolver

        self._a = a_upper
        self._solver = PyPardisoSolver(mtype=mtype)
        self._solver.set_statistical_info_off()
        self._solver.factorize(self._a)

    def solve(self, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        x = self._solver.solve(self._a, b)
        return x.reshape(b.shape)

    def free(self) -> None:
        solver = getattr(self, "_solver", None)
        if solver is not None:
            try:
                solver.free_memory(everything=True)
            finally:
                self._solver = None

    def __del__(self) -> None:
        self.free()


@dataclass(frozen=True)
class PardisoFactorizer:
    """Intel MKL PARDISO via ``pypardiso``: multithreaded and symmetry-aware.

    SPD matrices use a Cholesky-type factorization (matrix type 2), and
    indefinite ones use Bunch-Kaufman pivoting (type -2).  The thread count
    follows MKL's usual controls, e.g. ``MKL_NUM_THREADS``.
    """

    name: str = "pardiso"

    def __post_init__(self) -> None:
        if not pardiso_available():
            raise ImportError(
                "PardisoFactorizer needs pypardiso: `pip install pypardiso` "
                "(it brings Intel MKL). SuperLUFactorizer works without it."
            )

    def factorize(self, a: sp.spmatrix, *, definite: bool) -> Factorization:
        mtype = _REAL_SPD if definite else _REAL_SYMMETRIC_INDEFINITE
        return _PardisoFactorization(_upper_with_full_diagonal(a), mtype)


# -- threading ---------------------------------------------------------------


def openblas_single_threaded():
    """Context manager holding OpenBLAS to one thread, leaving MKL alone.

    The eigensolve alternates, every Lanczos step, between a triangular solve
    in the factorization backend and a small orthogonalization in numpy's
    OpenBLAS.  Both use OpenMP-style thread pools, and a pool that has just
    finished keeps its threads spinning -- so the two libraries spend the
    iteration stealing cores from each other.  Measured on the paper mesh:
    the PARDISO eigensolve takes 3.05 s with default threading and 0.53 s with
    OpenBLAS held to one thread.  Even SuperLU gains (2.72 s -> 2.08 s).  The
    OpenBLAS work here is memory-bound vector work that gains nothing from
    threads.

    Scoped rather than global, so importing this package never changes a host
    program's thread settings.  Only OpenBLAS is limited, because MKL also
    registers as a BLAS library and limiting it would make PARDISO
    single-threaded.  Without ``threadpoolctl`` this is a no-op; setting
    ``OPENBLAS_NUM_THREADS=1`` in the environment has the same effect.
    """
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return contextlib.nullcontext()
    # Select by internal API: a ``limits`` dict would be keyed by library
    # *prefix* (libopenblas, libscipy_openblas, ...), and an unknown key is
    # silently ignored -- which is how a first version of this did nothing.
    return ThreadpoolController().select(internal_api="openblas").limit(limits=1)


# -- selection ---------------------------------------------------------------


def pardiso_available() -> bool:
    return importlib.util.find_spec("pypardiso") is not None


def default_factorizer() -> Factorizer:
    """PARDISO if ``pypardiso`` is installed, otherwise SuperLU."""
    return PardisoFactorizer() if pardiso_available() else SuperLUFactorizer()


def make_factorizer(name: str) -> Factorizer:
    """Resolve ``"auto"``, ``"superlu"`` or ``"pardiso"`` to a backend."""
    if name == "auto":
        return default_factorizer()
    if name == "superlu":
        return SuperLUFactorizer()
    if name == "pardiso":
        return PardisoFactorizer()
    raise ValueError(f"unknown solver {name!r}; expected auto, superlu or pardiso")
