"""Topology optimization of Chladni patterns.

A reimplementation of Xue et al. (2024), "Topology optimization design of
Chladni patterns for vibration mode manipulability", for the single-mode case.

Every stage of the pipeline sits behind a small protocol -- see the table in
README.md -- so components can be replaced without touching the driver.
"""

from .analysis import ChladniAnalysis, ModalState
from .assembly import Assembler
from .bc import ClampedEdges, CornerClamps, DofSet, SimplySupportedEdges
from .config import (
    FilterConfig,
    MaterialConfig,
    MeshConfig,
    OptimizerConfig,
    ProblemConfig,
    TargetConfig,
    build_problem,
)
from .driver import History, OptimizationResult, TopologyOptimizer, WindowedConvergence
from .eigen import EigenSolution, ShiftInvertEigen
from .elements.mindlin_q4 import MindlinQ4
from .filters import (
    BetaContinuation,
    DensityFilter,
    ExponentialThreshold,
    IdentityFilter,
    NoContinuation,
)
from .formulations import ConstraintFormulation, MultiObjectiveFormulation
from .indicators import Average, Variance
from .levelset import (
    ContourReinitializer,
    LevelSetConfig,
    LevelSetUpdate,
    NodalToElement,
    WangHeaviside,
    build_levelset_problem,
)
from .material import SIMPTwoMaterial, SolidVoidErsatz
from .mesh import StructuredQuadMesh
from .optimizers.adamw import AdamW
from .optimizers.hamilton_jacobi import HamiltonJacobi
from .optimizers.mma import MMA
from .selection import SelectionVectors, build_selection_vectors, widths_in_elements
from .sensitivity import AdjointEigenvectorSensitivity
from .tracking import FixedIndexTracker, MACTracker

__all__ = [
    "AdamW", "AdjointEigenvectorSensitivity", "Assembler", "Average", "BetaContinuation",
    "ChladniAnalysis", "ClampedEdges", "ConstraintFormulation", "ContourReinitializer",
    "CornerClamps", "DensityFilter", "DofSet", "EigenSolution", "ExponentialThreshold",
    "FilterConfig", "FixedIndexTracker", "HamiltonJacobi", "History", "IdentityFilter",
    "LevelSetConfig", "LevelSetUpdate", "MACTracker", "MMA", "MaterialConfig",
    "MeshConfig", "MindlinQ4", "ModalState", "MultiObjectiveFormulation",
    "NoContinuation", "NodalToElement", "OptimizationResult", "OptimizerConfig",
    "ProblemConfig", "SIMPTwoMaterial", "SelectionVectors", "ShiftInvertEigen",
    "SimplySupportedEdges", "SolidVoidErsatz", "StructuredQuadMesh", "TargetConfig",
    "TopologyOptimizer", "Variance", "WangHeaviside", "WindowedConvergence",
    "build_levelset_problem", "build_problem", "build_selection_vectors",
    "widths_in_elements",
]
