"""Shared bounds and result types for search algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Scenario, SimulationResult
from ..simulator import minimum_avoidable_distance


PARAMETER_NAMES = (
    "ego_speed",
    "pedestrian_distance",
    "pedestrian_speed",
    "road_friction",
    "sensor_delay",
    "sensor_noise_std",
)

BOUNDS = np.asarray(
    [
        (8.0, 20.0),
        (8.0, 80.0),
        (0.0, 2.0),
        (0.35, 1.0),
        (0.0, 0.50),
        (0.0, 1.0),
    ],
    dtype=float,
)


@dataclass
class SearchRecord:
    evaluation: int
    scenario: Scenario
    result: SimulationResult


@dataclass
class SearchResult:
    method: str
    records: list[SearchRecord]
    best_record: SearchRecord
    archive_coverage: int = 0
    archive_signatures: tuple = ()

    @property
    def evaluations(self) -> int:
        return len(self.records)


def vector_to_scenario(vector: np.ndarray, seed: int) -> Scenario:
    """Map search coordinates to an avoidable scenario.

    The distance coordinate is rescaled from its full bounds onto the part a
    perfect controller could handle, so every method searches the same
    avoidable scenarios without rejected samples or boundary pile-up.
    """
    values = np.asarray(vector, dtype=float)
    if values.shape != (len(PARAMETER_NAMES),):
        raise ValueError("scenario vector must contain six parameters")
    low, high = BOUNDS[1]
    avoidable = max(low, minimum_avoidable_distance(values[0], values[2], values[3]))
    distance = avoidable + (values[1] - low) / (high - low) * (high - avoidable)
    return Scenario(
        ego_speed=float(values[0]),
        pedestrian_distance=float(distance),
        pedestrian_speed=float(values[2]),
        road_friction=float(values[3]),
        sensor_delay=float(values[4]),
        sensor_noise_std=float(values[5]),
        seed=int(seed),
    )


def random_vector(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])


def clip_vector(vector: np.ndarray) -> np.ndarray:
    return np.clip(vector, BOUNDS[:, 0], BOUNDS[:, 1])
