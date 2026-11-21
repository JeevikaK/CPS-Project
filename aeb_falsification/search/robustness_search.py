"""Bounded CMA-ES baseline using the shared severity-aware safety objective."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import cma
import numpy as np

from ..simulator import simulate
from .common import BOUNDS, SearchRecord, SearchResult, vector_to_scenario


def _minimize_cma(
    objective: Callable[[np.ndarray], float],
    dimension: int,
    budget: int,
    seed: int,
) -> None:
    """Evaluate exactly budget bounded points; objective owns result recording.

    Coordinates are normalized to [0, 1]. Complete generations update CMA's
    covariance and step size; a final partial generation is evaluated only.
    If CMA terminates early (e.g. flat fitness), restart within the same budget.
    """
    if not isinstance(budget, Integral) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    rng = np.random.default_rng(seed)
    evaluations = 0
    while evaluations < budget:
        strategy = cma.CMAEvolutionStrategy(
            rng.uniform(0.0, 1.0, dimension),
            0.2,
            {
                "bounds": [0.0, 1.0],
                # Use a local generator; pycma seed=0 otherwise means random.
                "seed": float('nan'),
                "randn": lambda *shape: rng.standard_normal(shape),
                "verbose": -9,
                "verb_log": 0,
            },
        )
        while evaluations < budget and not strategy.stop():
            count = min(strategy.popsize, budget - evaluations)
            candidates = strategy.ask(number=count)
            scores = [float(objective(np.asarray(point))) for point in candidates]
            if not np.all(np.isfinite(scores)):
                raise ValueError("CMA-ES objective must return finite scores")
            evaluations += count
            if count == strategy.popsize:
                strategy.tell(candidates, scores)


def robustness_guided_search(budget: int = 100, seed: int = 0) -> SearchResult:
    """Run bounded CMA-ES under an exact simulator-call budget.

    Preserve the existing objective: aggregate safety robustness before collision,
    and -10 - ego impact speed at collision. This is severity-aware falsification,
    not optimization of formal STL robustness alone. Scenario noise uses the same
    fixed seed throughout a run, matching the other existing search methods.
    """
    records: list[SearchRecord] = []

    def evaluate(unit_vector: np.ndarray) -> float:
        vector = BOUNDS[:, 0] + unit_vector * (BOUNDS[:, 1] - BOUNDS[:, 0])
        scenario = vector_to_scenario(vector, seed=seed)
        result = simulate(scenario)
        records.append(SearchRecord(len(records), scenario, result))
        return result.search_score

    _minimize_cma(evaluate, len(BOUNDS), budget, seed)
    best = min(records, key=lambda record: record.result.search_score)
    # Retain method identifier for compatibility with existing experiment CSVs.
    return SearchResult("robustness_guided", records, best)
