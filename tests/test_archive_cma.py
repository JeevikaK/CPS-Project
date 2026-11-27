"""Archive-guided CMA-ES, stall restarts, noise seeds, and the archive ablation."""
import csv
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

import aeb_falsification.search.cma_es as module
from aeb_falsification import Scenario, simulate
from aeb_falsification.ablation import ARMS, holm, paired_test, run_ablation
from aeb_falsification.archive import FailureArchive
from aeb_falsification.search import (
    archive_guided_cma_search,
    cma_search,
    diversity_aware_search,
    random_search,
    robustness_guided_search,
)
from aeb_falsification.search.cma_es import minimize_cma
from aeb_falsification.search.common import BOUNDS, PARAMETER_NAMES, noise_seed

SAFE = Scenario(8, 60, 2, 1, 0, 0, 1)


def coverage(result):
    archive = FailureArchive()
    for record in result.records:
        archive.record(record.result, record.scenario)
    return archive.coverage


class StallRestartTests(unittest.TestCase):
    def test_stuck_search_restarts_and_uses_exact_budget(self):
        # Values differ (so pycma's flat-fitness stop does not fire) but never
        # improve by the relative stall tolerance: the search is stuck.
        noise = np.random.default_rng(0)
        calls = []
        run = minimize_cma(lambda p: calls.append(p) or 1.0 + 1e-6 * noise.random(), 6, 200, 1)
        self.assertEqual(len(calls), 200)
        self.assertGreater(run.restarts, 2)
        self.assertIn("stall", run.restart_reasons)
        self.assertEqual(run.restart_evaluations[0], 0)

    def test_flat_objective_still_uses_exact_budget(self):
        calls = []
        run = minimize_cma(lambda p: calls.append(p) or 0.0, 6, 200, 1)
        self.assertEqual(len(calls), 200)
        self.assertGreater(run.restarts, 2)

    def test_restarts_can_be_disabled(self):
        run = minimize_cma(lambda p: 0.0, 6, 200, 1, stall_generations=None)
        self.assertNotIn("stall", run.restart_reasons)

    def test_converging_search_is_not_interrupted(self):
        run = minimize_cma(lambda p: float(np.sum((p - .3) ** 2)), 6, 300, 12)
        self.assertNotIn("stall", run.restart_reasons)

    def test_restart_centre_is_used(self):
        points = []

        def objective(point):
            points.append(point)
            return 0.0

        centre = np.full(6, .9)
        run = minimize_cma(objective, 6, 120, 3, restart_centre=lambda rng: centre)
        self.assertGreater(run.archive_seeded_restarts, 0)
        second = run.restart_evaluations[1]
        self.assertLess(np.linalg.norm(np.mean(points[second:second + 9], axis=0) - centre), .25)

    def test_invalid_settings(self):
        for kwargs in ({"stall_generations": 0}, {"sigma0": 0}, {"restart_sigma": 2}):
            with self.assertRaises(ValueError):
                minimize_cma(lambda p: 0.0, 6, 10, 0, **kwargs)
        for kwargs in ({"novelty_bonus": -1}, {"restart_exploration": 2}, {"noise": "sometimes"}):
            with self.assertRaises(ValueError):
                cma_search(10, 0, **kwargs)


class ArchiveGuidedCmaTests(unittest.TestCase):
    def test_exact_budget_bounds_and_reproducibility(self):
        for budget in (1, 9, 10, 37):
            with patch.object(module, "simulate", wraps=simulate) as simulator:
                first = archive_guided_cma_search(budget, seed=4)
            self.assertEqual(simulator.call_count, budget)
            self.assertEqual([r.evaluation for r in first.records], list(range(budget)))
            for record in first.records:
                values = np.array([asdict(record.scenario)[key] for key in PARAMETER_NAMES])
                self.assertTrue(np.all(values >= BOUNDS[:, 0] - 1e-9))
                self.assertTrue(np.all(values <= BOUNDS[:, 1] + 1e-9))
        second = archive_guided_cma_search(37, seed=4)
        self.assertEqual([r.scenario for r in first.records], [r.scenario for r in second.records])
        self.assertEqual(first.method, "archive_cma")
        self.assertEqual(first.archive_coverage, coverage(first))

    def test_without_archive_options_it_is_the_baseline(self):
        plain = cma_search(60, seed=2, method="robustness_guided")
        baseline = robustness_guided_search(60, seed=2)
        self.assertEqual([r.scenario for r in plain.records], [r.scenario for r in baseline.records])

    def test_novelty_bonus_rewards_only_new_categories(self):
        crash = Scenario(20, 8, 0, .35, .5, 0, 1)
        fitness = []

        def three_calls(objective, *args, **kwargs):
            fitness.extend(objective(np.full(6, .5)) for _ in range(3))
            return module.CmaRun()

        with patch.object(module, "simulate", return_value=simulate(crash)), \
                patch.object(module, "minimize_cma", side_effect=three_calls):
            cma_search(3, 0, novelty_bonus=30)
        score = simulate(crash).search_score
        self.assertEqual(fitness, [score - 30, score, score])

    def test_archive_restarts_are_seeded_from_failures(self):
        result = archive_guided_cma_search(300, seed=5, restart_exploration=0.0)
        self.assertGreater(result.optimizer.restarts, 0)
        self.assertEqual(result.optimizer.archive_seeded_restarts, result.optimizer.restarts)

    def test_no_failures_falls_back_to_uniform_restarts(self):
        with patch.object(module, "simulate", return_value=simulate(SAFE)):
            result = archive_guided_cma_search(200, seed=0)
        self.assertEqual(result.evaluations, 200)
        self.assertEqual(result.archive_coverage, 0)
        self.assertEqual(result.optimizer.archive_seeded_restarts, 0)

    def test_archive_guidance_covers_more_categories_than_plain_cma(self):
        seeds = range(7000, 7012)
        guided = np.mean([coverage(archive_guided_cma_search(100, s)) for s in seeds])
        plain = np.mean([coverage(robustness_guided_search(100, s)) for s in seeds])
        self.assertGreater(guided, plain)


class NoiseSeedTests(unittest.TestCase):
    def test_fixed_mode_reuses_the_run_seed(self):
        self.assertEqual({noise_seed(7, i, "fixed") for i in range(5)}, {7})
        result = random_search(5, seed=7, noise="fixed")
        self.assertEqual({r.scenario.seed for r in result.records}, {7})

    def test_per_evaluation_mode_is_distinct_and_reproducible(self):
        seeds = [noise_seed(7, i) for i in range(1000)]
        self.assertEqual(len(set(seeds)), 1000)
        self.assertEqual(seeds, [noise_seed(7, i) for i in range(1000)])
        self.assertNotEqual(noise_seed(7, 0), noise_seed(8, 0))

    def test_every_method_stores_its_noise_seed_for_replay(self):
        for search in (random_search, robustness_guided_search, diversity_aware_search,
                       archive_guided_cma_search):
            result = search(12, 3)
            self.assertEqual([r.scenario.seed for r in result.records], [noise_seed(3, i) for i in range(12)])
            record = result.records[-1]
            np.testing.assert_array_equal(simulate(record.scenario).trace.measured_distance,
                                          record.result.trace.measured_distance)

    def test_search_points_do_not_depend_on_noise_mode_for_random_search(self):
        fixed, fresh = random_search(10, 2, noise="fixed"), random_search(10, 2)
        self.assertEqual([r.scenario.ego_speed for r in fixed.records],
                         [r.scenario.ego_speed for r in fresh.records])


class AblationTests(unittest.TestCase):
    def test_one_command_writes_every_arm_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, comparisons = run_ablation(budget=12, seeds=(0, 1), output_dir=directory,
                                             noise_modes=("per_evaluation", "fixed"), reseeds=1)
            self.assertEqual(len(rows), len(ARMS) * 2 * 2)
            self.assertEqual({row.evaluations for row in rows}, {12})
            for name in ("summary.csv", "comparison.csv", "ablation.png", "results.md", "settings.json"):
                self.assertTrue(Path(directory, name).exists(), name)
            with Path(directory, "comparison.csv").open(encoding="utf-8") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual({row["versus"] for row in table}, {"cma"})
            self.assertNotIn("cma", {row["arm"] for row in table})

    def test_invalid_ablation_settings(self):
        for kwargs in ({"budget": 0}, {"seeds": ()}, {"arms": ("random",)}, {"noise_modes": ("x",)}):
            with self.assertRaises(ValueError):
                run_ablation(**{"budget": 5, "seeds": (0,), **kwargs})

    def test_paired_test_and_holm(self):
        rng = np.random.default_rng(0)
        mean, low, high, p = paired_test(np.full(20, 2.0) + rng.normal(0, .1, 20), rng)
        self.assertAlmostEqual(mean, 2.0, delta=.1)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertLess(p, .001)
        self.assertEqual(paired_test(np.zeros(10), rng)[3], 1.0)
        self.assertGreater(paired_test(rng.normal(0, 1, 20), rng)[3], .01)
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])


if __name__ == "__main__":
    unittest.main()
