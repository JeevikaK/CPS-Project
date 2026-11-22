"""Behavioral checks for optimizer quality, isolation, and budget accounting."""
import unittest
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
from aeb_falsification.search.robustness_search import _minimize_cma, robustness_guided_search
from aeb_falsification.search.common import BOUNDS, PARAMETER_NAMES
from aeb_falsification.simulator import simulate


class CmaSearchTests(unittest.TestCase):
    def test_exact_simulator_calls_and_bounds(self):
        for budget in (1, 8, 9, 10, 19, 100):
            with self.subTest(budget=budget), patch(
                'aeb_falsification.search.robustness_search.simulate', wraps=simulate
            ) as simulator:
                result = robustness_guided_search(budget, seed=0)
                self.assertEqual(simulator.call_count, budget)
                self.assertEqual([r.evaluation for r in result.records], list(range(budget)))
                for record in result.records:
                    values = np.array([asdict(record.scenario)[key] for key in PARAMETER_NAMES])
                    self.assertTrue(np.all(values >= BOUNDS[:, 0]))
                    self.assertTrue(np.all(values <= BOUNDS[:, 1]))
                self.assertEqual(result.best_record.result.search_score,
                                 min(r.result.search_score for r in result.records))

    def test_full_run_reproducibility_including_seed_zero(self):
        for seed in (0, 7):
            first = robustness_guided_search(35, seed)
            second = robustness_guided_search(35, seed)
            self.assertEqual([r.scenario for r in first.records], [r.scenario for r in second.records])
            self.assertEqual([r.result.search_score for r in first.records],
                             [r.result.search_score for r in second.records])
        other = robustness_guided_search(35, 8)
        self.assertNotEqual(first.records[0].scenario, other.records[0].scenario)

    def test_optimizer_reduces_known_quadratic(self):
        scores = []
        def objective(point):
            score = float(np.sum((point - .3) ** 2))
            scores.append(score)
            return score
        _minimize_cma(objective, 6, 500, 12)
        self.assertEqual(len(scores), 500)
        self.assertLess(min(scores[-100:]), 1e-4)
        self.assertLess(min(scores[-100:]), min(scores[:9]))

    def test_flat_objective_still_uses_exact_budget(self):
        scores = []
        _minimize_cma(lambda point: scores.append(point) or 0., 6, 120, 2)
        self.assertEqual(len(scores), 120)

    def test_does_not_modify_global_numpy_random_state(self):
        np.random.seed(42)
        expected = np.random.random(5)
        np.random.seed(42)
        robustness_guided_search(20, 0)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_invalid_budget(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                robustness_guided_search(value)
