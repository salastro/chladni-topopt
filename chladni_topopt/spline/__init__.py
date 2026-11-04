"""Spline-parametrised outline: the plate is the region inside one closed B-spline.

A boundary-geometry method without holes.  The design is a set of offsets of
B-spline control points along fixed directions; the outline can grow into the
void margin of a :class:`~chladni_topopt.domain.PlateInDomain` or shrink
inside it.  See :mod:`.config` for how it plugs into the analysis chain.
"""

from .config import SplineConfig, build_spline_problem
from .geometry import ClosedBSpline, rectangle_controls
from .mapping import SplineMap

__all__ = [
    "ClosedBSpline", "SplineConfig", "SplineMap", "build_spline_problem",
    "rectangle_controls",
]
