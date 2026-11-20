"""Shared bounds and result types for search algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Scenario, SimulationResult


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
        (8.0, 40.0),
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
    values = np.asarray(vector, dtype=float)
    if values.shape != (len(PARAMETER_NAMES),):
        raise ValueError("scenario vector must contain six parameters")
    return Scenario(
        ego_speed=float(values[0]),
        pedestrian_distance=float(values[1]),
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
