"""Scenario generation and JSON serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import Scenario
from .search.common import random_vector, vector_to_scenario


def sample_random_scenario(seed: int) -> Scenario:
    """Sample one bounded, reproducible scenario."""
    return vector_to_scenario(random_vector(np.random.default_rng(seed)), seed)


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(scenario), indent=2) + "\n")


def load_scenario(path: str | Path) -> Scenario:
    return Scenario(**json.loads(Path(path).read_text()))
