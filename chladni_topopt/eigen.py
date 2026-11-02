"""Generalised eigenvalue solver for ``(K - lambda M) phi = 0``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

from .bc import DofSet
from .linalg import Factorizer, default_factorizer, openblas_single_threaded


@dataclass(frozen=True)
class EigenSolution:
    """The lowest modes of one design.

    Attributes
    ----------
    values:
        ``(k,)`` eigenvalues ``lambda = omega**2``, ascending.
    reduced:
        ``(n_free, k)`` eigenvectors on the free DOFs, ``M``-orthonormal.
    full:
        ``(n_dof, k)`` the same vectors scattered back to global numbering,
        with zeros on the constrained DOFs.
    """

    values: np.ndarray
    reduced: np.ndarray
    full: np.ndarray

    @property
    def frequencies(self) -> np.ndarray:
        """Natural frequencies in Hz."""
        return np.sqrt(np.maximum(self.values, 0.0)) / (2.0 * np.pi)

    def mode(self, index: int) -> tuple[float, np.ndarray, np.ndarray]:
        """``(lambda, phi_reduced, phi_full)`` for one mode."""
        return self.values[index], self.reduced[:, index], self.full[:, index]


@runtime_checkable
class EigenSolver(Protocol):
    """Computes the lowest eigenpairs of a symmetric definite pencil."""

    n_modes: int

    def solve(self, k: sp.spmatrix, m: sp.spmatrix, dofs: DofSet) -> EigenSolution:
        ...


@dataclass
class ShiftInvertEigen:
    """ARPACK shift-invert around ``sigma``, returning the lowest modes.

    Shift-invert is what makes this tractable: the modes of interest are the
    lowest handful of a 36k-DOF pencil, and they are recomputed every
    optimization iteration.

    ``sigma = 0`` factorises ``K`` itself, which is well defined here because
    the four clamped corners leave no rigid-body motion.

    Two things keep the per-iteration cost down, and both were measured.  The
    factorization comes from a swappable :class:`~chladni_topopt.linalg.Factorizer`
    and is handed to ARPACK as ``OPinv``, so a faster backend (PARDISO) applies
    directly.  And with ``warm_start`` the Lanczos iteration starts from the
    previous call's modes: between optimization steps the design changes
    little, so the wanted subspace barely moves.  The warm start only changes
    how quickly ARPACK converges, not what it converges to.

    ``n_modes`` counts the lowest modes returned.  It must leave headroom above
    whichever mode is being shaped, because mode tracking can only follow a
    mode that stays inside the computed window.  That check lives in
    :func:`~chladni_topopt.config.build_problem`, which knows the targets; the
    solver itself does not.
    """

    n_modes: int = 8
    sigma: float = 0.0
    tol: float = 0.0
    factorizer: Factorizer = field(default_factory=default_factorizer)
    warm_start: bool = True
    _previous: np.ndarray | None = field(default=None, init=False, repr=False)

    def solve(self, k: sp.spmatrix, m: sp.spmatrix, dofs: DofSet) -> EigenSolution:
        shifted = k if self.sigma == 0.0 else k - self.sigma * m
        # K - sigma M stays positive definite for any shift at or below zero.
        fact = self.factorizer.factorize(shifted, definite=self.sigma <= 0.0)
        opinv = LinearOperator(k.shape, matvec=fact.solve, dtype=float)

        v0 = None
        if (
            self.warm_start
            and self._previous is not None
            and self._previous.shape[0] == k.shape[0]
        ):
            # One starting vector with a component along every wanted mode.
            v0 = self._previous.sum(axis=1)

        with openblas_single_threaded():
            values, vectors = eigsh(
                k.tocsc(),
                k=self.n_modes,
                M=m.tocsc(),
                sigma=self.sigma,
                which="LM",
                tol=self.tol,
                OPinv=opinv,
                v0=v0,
            )
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]

        # ARPACK returns M-orthonormal vectors in shift-invert mode, but the
        # normalisation is load-bearing for the adjoint (phi^T M phi = 1), so
        # enforce it rather than assume it -- all the more with a user-supplied
        # OPinv.
        norms = np.sqrt(np.einsum("ik,ik->k", vectors, m @ vectors))
        vectors = vectors / norms

        vectors = _canonical_sign(vectors)
        if self.warm_start:
            self._previous = vectors
        return EigenSolution(
            values=values, reduced=vectors, full=dofs.expand(vectors)
        )


def _canonical_sign(vectors: np.ndarray) -> np.ndarray:
    """Fix the sign of each eigenvector so plots and MAC traces are stable.

    The indicators all use ``phi o phi`` and are sign-blind, so this only
    affects presentation and the readability of mode-tracking diagnostics.
    """
    peak = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[peak, np.arange(vectors.shape[1])])
    signs[signs == 0] = 1.0
    return vectors * signs
