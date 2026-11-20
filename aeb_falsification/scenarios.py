"""Scenario generation and JSON serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import Scenario


def sample_random_scenario(seed: int) -> Scenario:
    """Sample one bounded, reproducible scenario."""
    rng = np.random.default_rng(seed)
    return Scenario(
        ego_speed=float(rng.uniform(8.0, 20.0)),
        pedestrian_distance=float(rng.uniform(8.0, 40.0)),
        pedestrian_speed=float(rng.uniform(0.0, 2.0)),
        road_friction=float(rng.uniform(0.35, 1.0)),
        sensor_delay=float(rng.uniform(0.0, 0.50)),
        sensor_noise_std=float(rng.uniform(0.0, 1.0)),
        seed=int(seed),
    )


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(scenario), indent=2) + "\n")


def load_scenario(path: str | Path) -> Scenario:
    return Scenario(**json.loads(Path(path).read_text()))
