"""Behavioral checks for archive-guided diversity search."""
import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

import aeb_falsification.search.diversity_search as module
from aeb_falsification import Scenario, simulate
from aeb_falsification.archive import FailureArchive
from aeb_falsification.search import diversity_aware_search, random_search
from aeb_falsification.search.common import BOUNDS, PARAMETER_NAMES, random_vector


def coverage(result):
    archive = FailureArchive()
    for record in result.records:
        archive.record(record.result, record.scenario)
    return archive.coverage


class DiversitySearchTests(unittest.TestCase):
    def test_exact_budget_and_bounds(self):
        for budget in (1, 9, 10, 11, 60):
            with patch.object(module, "simulate", wraps=simulate) as simulator:
                result = diversity_aware_search(budget, seed=0)
            self.assertEqual(simulator.call_count, budget)
            self.assertEqual([r.evaluation for r in result.records], list(range(budget)))
            for record in result.records:
                values = np.array([asdict(record.scenario)[key] for key in PARAMETER_NAMES])
                self.assertTrue(np.all(values >= BOUNDS[:, 0] - 1e-9))
                self.assertTrue(np.all(values <= BOUNDS[:, 1] + 1e-9))

    def test_reproducible_and_reported_coverage_matches_records(self):
        first, second = (diversity_aware_search(60, seed=5) for _ in range(2))
        self.assertEqual([r.scenario for r in first.records], [r.scenario for r in second.records])
        self.assertEqual(first.archive_coverage, coverage(first))

    def test_most_proposals_come_from_the_archive(self):
        with patch.object(module, "random_vector", wraps=random_vector) as sampler:
            diversity_aware_search(100, seed=0)
        self.assertLess(sampler.call_count, 45)

    def test_falls_back_to_random_when_nothing_fails(self):
        safe = simulate(Scenario(8, 60, 2, 1, 0, 0, 1))
        with patch.object(module, "simulate", return_value=safe):
            result = diversity_aware_search(30, seed=0)
        self.assertEqual(result.evaluations, 30)
        self.assertEqual(result.archive_coverage, 0)

    def test_covers_more_failure_categories_than_random(self):
        seeds = range(15)
        guided = np.mean([coverage(diversity_aware_search(100, s)) for s in seeds])
        uniform = np.mean([coverage(random_search(100, s)) for s in seeds])
        self.assertGreater(guided, uniform)

    def test_invalid_settings(self):
        for settings in ({"budget": 0}, {"exploration_probability": 1.5},
                         {"mutation_std": 0.0}, {"initial_random": -1}):
            with self.assertRaises(ValueError):
                diversity_aware_search(**settings)


if __name__ == "__main__":
    unittest.main()
