"""Uniform random falsification baseline."""

from __future__ import annotations

import numpy as np

from ..simulator import simulate
from .common import SearchRecord, SearchResult, random_vector, vector_to_scenario


def random_search(budget: int = 100, seed: int = 0) -> SearchResult:
    """Evaluate uniformly sampled scenarios under a fixed simulation budget."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    records: list[SearchRecord] = []
    for evaluation in range(budget):
        scenario = vector_to_scenario(random_vector(rng), seed=seed)
        result = simulate(scenario)
        records.append(SearchRecord(evaluation, scenario, result))
    best = min(records, key=lambda record: record.result.search_score)
    return SearchResult("random", records, best)
