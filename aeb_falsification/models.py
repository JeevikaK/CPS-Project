"""Public data structures for the AEB falsification prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ControllerMode(str, Enum):
    CRUISING = "cruising"
    WARNING = "warning"
    EMERGENCY_BRAKING = "emergency_braking"


@dataclass(frozen=True)
class Scenario:
    """One reproducible test scenario.

    The one-dimensional road places the pedestrian initially ahead of the ego
    vehicle. Pedestrian speed is longitudinal in this first prototype.
    """

    ego_speed: float
    pedestrian_distance: float
    pedestrian_speed: float
    road_friction: float
    sensor_delay: float
    sensor_noise_std: float
    seed: int


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    horizon: float = 10.0
    target_speed: float = 15.0
    warning_ttc: float = 3.0
    emergency_ttc: float = 1.5
    minimum_separation: float = 2.0
    collision_distance: float = 0.0
    maximum_acceleration: float = 2.0
    normal_braking: float = -4.0
    emergency_braking: float = -8.0
    maximum_sensor_delay: float = 0.50


@dataclass(frozen=True)
class ControllerConfig:
    target_speed: float = 15.0
    warning_ttc: float = 3.0
    emergency_ttc: float = 1.5
    minimum_separation: float = 2.0
    maximum_acceleration: float = 2.0
    normal_braking: float = -4.0
    emergency_braking: float = -8.0
    friction_factor: float = 1.0


@dataclass
class Trace:
    time: np.ndarray
    ego_position: np.ndarray
    ego_speed: np.ndarray
    pedestrian_position: np.ndarray
    measured_distance: np.ndarray
    true_distance: np.ndarray
    ttc: np.ndarray
    brake_command: np.ndarray
    controller_mode: np.ndarray
    collision: bool
    impact_speed: float


@dataclass(frozen=True)
class SafetyMetrics:
    distance_robustness: float
    separation_robustness: float
    ttc_robustness: float
    overall_robustness: float


@dataclass
class SimulationResult:
    trace: Trace
    min_distance: float
    min_ttc: float
    collision: bool
    failure_type: Optional[str]
    severity: float
    metrics: SafetyMetrics
    impact_speed: float

    @property
    def search_score(self) -> float:
        """Lower is worse; separate collision severity from STL robustness."""
        if self.collision:
            # Keep collision cases distinguishable even though formal distance
            # robustness bottoms out at the collision boundary.
            return -10.0 - self.impact_speed
        return self.metrics.overall_robustness
