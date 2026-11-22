"""Diversity-aware falsification guided by the failure archive."""

from __future__ import annotations

import numpy as np

from ..archive import FailureArchive
from ..simulator import simulate
from .common import (
    BOUNDS,
    SearchRecord,
    SearchResult,
    clip_vector,
    random_vector,
    vector_to_scenario,
)


def diversity_aware_search(
    budget: int = 100,
    seed: int = 0,
    exploration_probability: float = 0.2,
    mutation_std: float = 0.2,
    initial_random: int = 10,
) -> SearchResult:
    """Mutate archived failures to reach new failure categories.

    After ``initial_random`` uniform samples, each proposal perturbs the most
    severe known failure of a uniformly chosen category, or is a fresh uniform
    sample with ``exploration_probability``. Choosing among categories, rather
    than following one current point, keeps the search from settling on a
    single failure. ``mutation_std`` is a fraction of each parameter's range.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration_probability must be between 0 and 1")
    if mutation_std <= 0.0:
        raise ValueError("mutation_std must be positive")
    if initial_random < 0:
        raise ValueError("initial_random must be non-negative")

    rng = np.random.default_rng(seed)
    archive = FailureArchive()
    records: list[SearchRecord] = []
    # Search coordinates of the most severe failure in each category.
    elites: dict = {}
    scales = (BOUNDS[:, 1] - BOUNDS[:, 0]) * mutation_std

    for evaluation in range(budget):
        if evaluation < initial_random or not elites or rng.random() < exploration_probability:
            vector = random_vector(rng)
        else:
            parents = list(elites.values())
            _, parent = parents[rng.integers(len(parents))]
            vector = clip_vector(parent + rng.normal(0.0, scales))
        scenario = vector_to_scenario(vector, seed=seed)
        result = simulate(scenario)
        signature, _ = archive.record(result, scenario)
        records.append(SearchRecord(evaluation, scenario, result))
        if result.failure_type is not None and (
            signature not in elites or result.search_score < elites[signature][0]
        ):
            elites[signature] = (result.search_score, vector)

    best = min(records, key=lambda record: record.result.search_score)
    return SearchResult(
        method="diversity_aware",
        records=records,
        best_record=best,
        archive_coverage=archive.coverage,
        archive_signatures=archive.signatures,
    )
