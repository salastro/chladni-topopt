"""Shared fixtures: a small but complete problem the tests can drive end to end."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chladni_topopt.analysis import ChladniAnalysis
from chladni_topopt.assembly import Assembler
from chladni_topopt.bc import CornerClamps, DofSet
from chladni_topopt.eigen import ShiftInvertEigen
from chladni_topopt.elements.mindlin_q4 import MindlinQ4
from chladni_topopt.filters import DensityFilter, ExponentialThreshold
from chladni_topopt.formulations import ConstraintFormulation
from chladni_topopt.linalg import (
    PardisoFactorizer,
    SuperLUFactorizer,
    default_factorizer,
    pardiso_available,
)
from chladni_topopt.material import SIMPTwoMaterial
from chladni_topopt.mesh import StructuredQuadMesh
from chladni_topopt.selection import build_selection_vectors
from chladni_topopt.tracking import FixedIndexTracker
from examples.shapes import letter_d

#: Small enough for a few hundred eigensolves, large enough to be a real problem.
NX, NY = 20, 24


@pytest.fixture(scope="session")
def mesh() -> StructuredQuadMesh:
    return StructuredQuadMesh(2.0, 2.4, NX, NY)


@pytest.fixture(scope="session")
def element() -> MindlinQ4:
    return MindlinQ4(thickness=0.2, poisson=0.3)


@pytest.fixture(scope="session")
def dofs(mesh) -> DofSet:
    return DofSet.from_mesh(mesh, CornerClamps(0.1))


@pytest.fixture(scope="session")
def assembler(mesh, element, dofs) -> Assembler:
    return Assembler(mesh, element, dofs)


@pytest.fixture(scope="session")
def selection(mesh, dofs):
    unit = mesh.dx
    return build_selection_vectors(
        mesh,
        dofs,
        letter_d(1.0, 1.2, 1.0, 1.2),
        band=1.0 * unit,
        gap=1.0 * unit,
        surround=4.0 * unit,
    )


def make_analysis(
    mesh,
    element,
    dofs,
    assembler,
    selection,
    formulation=None,
    beta: float = 1.0,
    mode_index: int = 4,
    density_filter=None,
    factorizer=None,
) -> ChladniAnalysis:
    """A complete analysis chain with a non-adaptive tracker.

    Mode tracking is deliberately fixed here: a MAC tracker mutates its
    reference on every call, which would make the repeated evaluations of a
    finite-difference check depend on their own order.
    """
    factorizer = factorizer or default_factorizer()
    return ChladniAnalysis(
        assembler=assembler,
        dofs=dofs,
        material=SIMPTwoMaterial(e1=68.5e9, e2=1.0e9, rho1=2700.0, rho2=1180.0),
        filter=density_filter
        or DensityFilter(rmin=2.0 * mesh.dx, centroids=mesh.elem_centroids()),
        projection=ExponentialThreshold(eta=0.5, beta=beta),
        eigen_solver=ShiftInvertEigen(n_modes=8, factorizer=factorizer),
        factorizer=factorizer,
        formulation=formulation or ConstraintFormulation(),
        selections=[selection],
        trackers=[FixedIndexTracker(index=mode_index)],
        weights=[1.0],
    )


@pytest.fixture
def analysis(mesh, element, dofs, assembler, selection):
    return make_analysis(mesh, element, dofs, assembler, selection)


@pytest.fixture
def design(mesh) -> np.ndarray:
    """A non-uniform starting design, so no symmetry hides a sign error."""
    return np.random.default_rng(0).uniform(0.35, 0.65, mesh.n_elem)


#: Every factorization backend usable in this environment.  Tests that depend
#: on solver accuracy run once per backend, so a PARDISO install is exercised
#: by the same finite-difference checks as SuperLU.
FACTORIZERS = [SuperLUFactorizer()] + ([PardisoFactorizer()] if pardiso_available() else [])


@pytest.fixture(params=FACTORIZERS, ids=lambda f: f.name)
def factorizer(request):
    return request.param
