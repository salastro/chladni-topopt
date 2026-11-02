"""Level-set (boundary-geometry) parameterisation, after Wang, Wang & Guo (2003).

The density method designs *how much* material each element holds.  Here the
design is the plate's boundary itself: the zero contour of a nodal function
``Phi``, positive in material and negative in holes, moved by a
Hamilton-Jacobi equation.  Each piece plugs into an existing slot of the
analysis chain -- see :mod:`.config` for the mapping.
"""

from .config import LevelSetConfig, build_levelset_problem
from .heaviside import WangHeaviside
from .initial import frozen_nodes, perforated
from .mapping import NodalToElement
from .reinit import ContourReinitializer, Reinitializer
from .update import LevelSetUpdate

__all__ = [
    "ContourReinitializer", "LevelSetConfig", "LevelSetUpdate", "NodalToElement",
    "Reinitializer", "WangHeaviside", "build_levelset_problem", "frozen_nodes",
    "perforated",
]
