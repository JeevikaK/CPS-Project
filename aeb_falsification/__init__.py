"""Dependency-light AEB falsification prototype."""

from .models import (
    ControllerConfig,
    ControllerMode,
    Scenario,
    SimulationConfig,
    SimulationResult,
    Trace,
)
from .simulator import simulate

__all__ = [
    "ControllerConfig",
    "ControllerMode",
    "Scenario",
    "SimulationConfig",
    "SimulationResult",
    "Trace",
    "simulate",
]
