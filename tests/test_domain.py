"""A plate inside a larger design domain, and the formulation without a volume cap."""

from __future__ import annotations

import numpy as np
import pytest

from chladni_topopt.bc import CornerClamps
from chladni_topopt.domain import PlateInDomain
from chladni_topopt.formulations import ConstraintFormulation
from tests.conftest import make_analysis
from tests.test_sensitivity import _fd_vs_adjoint, _probe_elements


def test_clamps_sit_at_the_plate_corners_not_the_domain_corners():
    domain = PlateInDomain(plate=(2.0, 2.4), margin=0.3, element=0.1)
    mesh = domain.mesh().build()
    nodes = np.unique(domain.clamps().fixed_dofs(mesh) // mesh.ndof_per_node)
    xy = mesh.node_coords()[nodes]
    # 0.2 m squares at (0.3, 0.3) .. (2.3, 2.7): 3 x 3 nodes each on a 0.1 m grid.
    assert len(nodes) == 4 * 9
    assert xy[:, 0].min() == pytest.approx(0.3) and xy[:, 0].max() == pytest.approx(2.3)
    assert xy[:, 1].min() == pytest.approx(0.3) and xy[:, 1].max() == pytest.approx(2.7)


def test_zero_margin_reproduces_the_paper_clamps():
    domain = PlateInDomain(plate=(2.0, 2.4), margin=0.0, element=0.1)
    mesh = domain.mesh().build()
    assert np.array_equal(
        np.sort(domain.clamps().fixed_dofs(mesh)), np.sort(CornerClamps().fixed_dofs(mesh))
    )


def test_shift_and_signed_distance():
    domain = PlateInDomain(plate=(2.0, 2.4), margin=0.3)
    assert np.allclose(domain.shift([np.array([[0.0, 0.0]])])[0], [[0.3, 0.3]])
    points = np.array([[1.3, 1.5], [0.3, 1.0], [0.0, 1.0], [2.6, 3.0]])
    sd = domain.plate_sdf(points)
    assert sd[0] == pytest.approx(1.0)                 # centre: 1 m from the long sides
    assert sd[1] == pytest.approx(0.0)                 # on the left edge
    assert sd[2] == pytest.approx(-0.3)                # in the margin
    assert sd[3] == pytest.approx(-np.hypot(0.3, 0.3))  # beyond a corner


def test_no_volume_cap_drops_the_constraint(mesh, element, dofs, assembler, selection, design):
    formulation = ConstraintFormulation(vstar=None)
    assert formulation.constraint_names == ("avg_shape", "var_surround")
    analysis = make_analysis(mesh, element, dofs, assembler, selection, formulation=formulation)
    capped = make_analysis(mesh, element, dofs, assembler, selection)
    a = analysis.responses(analysis.solve(design))
    b = capped.responses(capped.solve(design))
    assert len(a.values) == 3 and np.allclose(a.values, b.values[:3])

    probes = _probe_elements(mesh.n_elem, dofs, assembler)[:6]
    fd, adjoint, names = _fd_vs_adjoint(analysis, design, probes)
    for row, name in enumerate(names):
        scale = max(np.abs(adjoint[row]).max(), np.abs(fd[row]).max())
        assert np.abs(fd[row] - adjoint[row]).max() / scale < 1e-4, name
