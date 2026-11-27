"""Robustness-guided baseline: plain CMA-ES with stall restarts, no archive."""

from __future__ import annotations

from .cma_es import cma_search, minimize_cma
from .common import DEFAULT_NOISE_MODE, SearchResult

# Kept for callers of the earlier private name.
_minimize_cma = minimize_cma


def robustness_guided_search(
    budget: int = 100,
    seed: int = 0,
    *,
    stall_generations: int | None = 4,
    noise: str = DEFAULT_NOISE_MODE,
) -> SearchResult:
    """Run bounded CMA-ES under an exact simulator-call budget.

    The objective is aggregate safety robustness before collision, and
    -10 - ego impact speed at collision (severity-aware falsification, not formal
    STL robustness alone). This is the no-archive arm of the archive ablation.
    """
    # Retain method identifier for compatibility with existing experiment CSVs.
    return cma_search(budget, seed, stall_generations=stall_generations, noise=noise,
                      method="robustness_guided")
