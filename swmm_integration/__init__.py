"""PySWMM integration layer for the Cariari hydrology workflow."""

from .engine import PySwmmUnavailableError, SwmmSimulationService
from .hydrographs import generate_hydrographs
from .rainfall import build_rainfall_intervals
from .validation import validate_inp_file

__all__ = [
    "PySwmmUnavailableError",
    "SwmmSimulationService",
    "build_rainfall_intervals",
    "generate_hydrographs",
    "validate_inp_file",
]
