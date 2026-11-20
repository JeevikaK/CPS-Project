"""A lightweight robustness-guided search baseline."""

from __future__ import annotations

import numpy as np

from ..simulator import simulate
from .common import (
    BOUNDS,
    SearchRecord,
    SearchResult,
    clip_vector,
    random_vector,
    vector_to_scenario,
)


def robustness_guided_search(budget: int = 100, seed: int = 0) -> SearchResult:
    """Search for low STL robustness using seeded adaptive local mutations.

    This is deliberately a small dependency-free baseline. The diversity-aware
    search can later reuse this interface and add an archive/novelty objective.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    rng = np.random.default_rng(seed)
    records: list[SearchRecord] = []

    current = random_vector(rng)
    current_result = simulate(vector_to_scenario(current, seed=seed))
    records.append(SearchRecord(0, vector_to_scenario(current, seed=seed), current_result))

    scales = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.15
    for evaluation in range(1, budget):
        progress = evaluation / max(1, budget - 1)
        proposal = clip_vector(current + rng.normal(0.0, scales * (1.0 - 0.8 * progress)))
        proposal_scenario = vector_to_scenario(proposal, seed=seed)
        proposal_result = simulate(proposal_scenario)
        proposal_record = SearchRecord(evaluation, proposal_scenario, proposal_result)
        records.append(proposal_record)

        if proposal_result.search_score < current_result.search_score:
            current = proposal
            current_result = proposal_result

    best = min(records, key=lambda record: record.result.search_score)
    return SearchResult("robustness_guided", records, best)
