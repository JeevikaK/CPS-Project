"""CMA-ES with stall restarts, optionally guided by the failure archive.

One optimizer serves both arms of the archive ablation: ``cma_search`` with
``novelty_bonus=0`` and ``archive_restarts=False`` is plain CMA-ES (the
robustness-guided baseline); turning either option on adds archive guidance and
changes nothing else.

Archive guidance has two parts:

* Novelty fitness. CMA-ES minimizes ``search_score - novelty_bonus`` when a
  simulation lands in a failure category the archive has not seen, and plain
  ``search_score`` otherwise. Selection is rank-based, so a bonus larger than
  the score range (about 30) puts every new category ahead of every known one.
* Archive restarts. When a run is stuck, the next restart is centred on the
  stored worst failure of the least-visited archive category (with probability
  ``1 - restart_exploration``) instead of a uniform random point.

A run is stuck when the best generation-median fitness has not improved by the
relative amount ``stall_tolerance`` (times its magnitude) for ``stall_generations``
complete generations, when every coordinate's step
size is below ``min_step`` (converged), or when pycma itself stops.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral

import cma
import numpy as np

from ..archive import FailureArchive
from ..simulator import simulate
from .common import BOUNDS, DEFAULT_NOISE_MODE, SearchRecord, SearchResult, noise_seed, vector_to_scenario

RestartCentre = Callable[[np.random.Generator], "np.ndarray | None"]


@dataclass
class CmaRun:
    """What the optimizer did, for reporting."""

    restart_evaluations: list[int] = field(default_factory=list)
    restart_reasons: list[str] = field(default_factory=list)
    archive_seeded_restarts: int = 0

    @property
    def restarts(self) -> int:
        """Restarts after the first start."""
        return max(0, len(self.restart_evaluations) - 1)


def minimize_cma(
    objective: Callable[[np.ndarray], float],
    dimension: int,
    budget: int,
    seed: int,
    *,
    sigma0: float = 0.2,
    restart_sigma: float = 0.15,
    stall_generations: int | None = 4,
    stall_tolerance: float = 1e-3,
    min_step: float = 1e-3,
    restart_centre: RestartCentre | None = None,
) -> CmaRun:
    """Evaluate exactly ``budget`` points in [0, 1]^dimension with restarting CMA-ES.

    Complete generations update the mean, covariance, and step size; a final
    partial generation is evaluated only. ``stall_generations=None`` disables
    stall restarts (only pycma's own stop and convergence restart).
    ``restart_centre`` may return a restart mean, or ``None`` for a uniform one.
    """
    if not isinstance(budget, Integral) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")
    if stall_generations is not None and stall_generations < 1:
        raise ValueError("stall_generations must be positive or None")
    if not 0 < sigma0 <= 1 or not 0 < restart_sigma <= 1:
        raise ValueError("step sizes must be in (0, 1]")
    rng = np.random.default_rng(seed)
    run = CmaRun()
    evaluations = 0
    while evaluations < budget:
        centre = None
        if run.restart_evaluations and restart_centre is not None:
            centre = restart_centre(rng)
        if centre is None:
            mean, sigma = rng.uniform(0.0, 1.0, dimension), sigma0
        else:
            mean, sigma = np.clip(np.asarray(centre, dtype=float), 0.0, 1.0), restart_sigma
            run.archive_seeded_restarts += 1
        run.restart_evaluations.append(evaluations)
        strategy = cma.CMAEvolutionStrategy(
            mean,
            sigma,
            {
                "bounds": [0.0, 1.0],
                # Use a local generator; pycma seed=0 otherwise means random.
                "seed": float("nan"),
                "randn": lambda *shape: rng.standard_normal(shape),
                "verbose": -9,
                "verb_log": 0,
            },
        )
        best, stalled, reason = np.inf, 0, "budget"
        while evaluations < budget:
            if strategy.stop():
                reason = "pycma_stop"
                break
            count = min(strategy.popsize, budget - evaluations)
            candidates = strategy.ask(number=count)
            scores = [float(objective(np.asarray(point))) for point in candidates]
            if not np.all(np.isfinite(scores)):
                raise ValueError("CMA-ES objective must return finite scores")
            evaluations += count
            if count < strategy.popsize:
                break
            strategy.tell(candidates, scores)
            # Median, not best: a converging search keeps lowering its median even
            # when no single sample beats the best so far.
            median = float(np.median(scores))
            if not np.isfinite(best) or median < best - stall_tolerance * abs(best):
                best, stalled = median, 0
            else:
                stalled += 1
            if stall_generations is not None and stalled >= stall_generations:
                reason = "stall"
                break
            if float(np.max(strategy.stds)) < min_step:
                reason = "converged"
                break
        run.restart_reasons.append(reason)
    return run


def cma_search(
    budget: int = 100,
    seed: int = 0,
    *,
    novelty_bonus: float = 0.0,
    archive_restarts: bool = False,
    restart_exploration: float = 0.3,
    stall_generations: int | None = 4,
    noise: str = DEFAULT_NOISE_MODE,
    method: str = "cma_es",
) -> SearchResult:
    """CMA-ES over the scenario space; archive guidance is off by default."""
    if novelty_bonus < 0:
        raise ValueError("novelty_bonus must be non-negative")
    if not 0 <= restart_exploration <= 1:
        raise ValueError("restart_exploration must be between 0 and 1")
    noise_seed(seed, 0, noise)  # validate the mode before any simulation
    records: list[SearchRecord] = []
    archive = FailureArchive()
    visits: Counter = Counter()
    elites: dict = {}  # signature -> (search score, unit coordinates)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]

    def evaluate(unit: np.ndarray) -> float:
        scenario = vector_to_scenario(BOUNDS[:, 0] + unit * span, seed=noise_seed(seed, len(records), noise))
        result = simulate(scenario)
        records.append(SearchRecord(len(records), scenario, result))
        signature, is_new = archive.record(result, scenario)
        if result.failure_type is None:
            return result.search_score
        visits[signature] += 1
        if signature not in elites or result.search_score < elites[signature][0]:
            elites[signature] = (result.search_score, np.array(unit))
        return result.search_score - novelty_bonus * float(is_new)

    def restart_centre(rng: np.random.Generator):
        if not elites or rng.random() < restart_exploration:
            return None
        fewest = min(visits[s] for s in elites)
        candidates = [s for s in archive.signatures if s in elites and visits[s] == fewest]
        return elites[candidates[rng.integers(len(candidates))]][1]

    run = minimize_cma(
        evaluate,
        len(BOUNDS),
        budget,
        seed,
        stall_generations=stall_generations,
        restart_centre=restart_centre if archive_restarts else None,
    )
    best = min(records, key=lambda record: record.result.search_score)
    return SearchResult(method, records, best, archive.coverage, archive.signatures, run)


def archive_guided_cma_search(
    budget: int = 100,
    seed: int = 0,
    *,
    novelty_bonus: float = 30.0,
    archive_restarts: bool = True,
    restart_exploration: float = 0.3,
    stall_generations: int | None = 4,
    noise: str = DEFAULT_NOISE_MODE,
) -> SearchResult:
    """CMA-ES with novelty fitness and archive-seeded restarts."""
    return cma_search(
        budget,
        seed,
        novelty_bonus=novelty_bonus,
        archive_restarts=archive_restarts,
        restart_exploration=restart_exploration,
        stall_generations=stall_generations,
        noise=noise,
        method="archive_cma",
    )
