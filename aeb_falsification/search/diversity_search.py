"""Diversity-aware falsification using the failure archive."""

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
    novelty_weight: float = 15.0,
    exploration_probability: float = 0.65,
) -> SearchResult:
    """Search for severe failures while rewarding new archive cells."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if novelty_weight < 0.0:
        raise ValueError("novelty_weight must be non-negative")
    if not 0.0 <= exploration_probability <= 1.0:
        raise ValueError("exploration_probability must be between 0 and 1")

    rng = np.random.default_rng(seed)
    archive = FailureArchive()
    records: list[SearchRecord] = []
    current = random_vector(rng)
    current_scenario = vector_to_scenario(current, seed=seed)
    current_result = simulate(current_scenario)
    _, is_new = archive.record(current_result, current_scenario)
    records.append(SearchRecord(0, current_scenario, current_result))
    current_score = current_result.search_score - novelty_weight * float(is_new)

    scales = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.15
    repeated_cells = 0
    for evaluation in range(1, budget):
        progress = evaluation / max(1, budget - 1)
        adaptive_probability = min(
            0.95,
            exploration_probability + 0.10 * min(repeated_cells, 3),
        )
        if rng.random() < adaptive_probability:
            # Periodic global exploration prevents the archive from becoming
            # a record of only the neighborhood around the first collision.
            proposal = random_vector(rng)
        else:
            proposal = clip_vector(current + rng.normal(0.0, scales * (1.0 - 0.8 * progress)))
        proposal_scenario = vector_to_scenario(proposal, seed=seed)
        proposal_result = simulate(proposal_scenario)
        proposal_signature, proposal_is_new = archive.record(proposal_result, proposal_scenario)
        if not proposal_is_new:
            repeated_cells += 1
        else:
            repeated_cells = 0
        proposal_score = proposal_result.search_score - novelty_weight * float(proposal_is_new)
        records.append(SearchRecord(evaluation, proposal_scenario, proposal_result))

        if proposal_score < current_score:
            current = proposal
            current_result = proposal_result
            current_score = proposal_score

    best = min(records, key=lambda record: record.result.search_score)
    return SearchResult(
        method="diversity_aware",
        records=records,
        best_record=best,
        archive_coverage=archive.coverage,
        archive_signatures=archive.signatures,
    )
