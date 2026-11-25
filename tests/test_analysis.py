import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aeb_falsification.analysis import (
    bootstrap_mean_ci,
    holm_adjust,
    load_runs,
    run_analysis,
    wilcoxon_signed_rank,
)

FIELDS = (
    "method,seed,evaluations,best_search_score,best_formal_robustness,best_failure_type,"
    "best_impact_speed,algorithm_archive_coverage,observed_failure_coverage,collisions,"
    "violations,first_failure_evaluation,worst_impact_speed_kmh,collision_rate,violation_rate"
).split(",")


def write_results(folder: Path, budget: int, seeds, coverage_by_method, with_curve=True):
    folder.mkdir(parents=True, exist_ok=True)
    rows, curve = [], []
    for method, offset in coverage_by_method.items():
        for seed in seeds:
            coverage = offset + seed % 3
            rows.append(dict(
                method=method, seed=seed, evaluations=budget, best_search_score=-1,
                best_formal_robustness=-2, best_failure_type="collision", best_impact_speed=10,
                algorithm_archive_coverage=coverage, observed_failure_coverage=coverage,
                collisions=seed, violations=seed + 1,
                first_failure_evaluation=-1 if seed == 0 else 2,
                worst_impact_speed_kmh=30 + seed, collision_rate=seed / budget,
                violation_rate=(seed + 1) / budget,
            ))
            for evaluation in range(1, budget + 1):
                curve.append(dict(method=method, seed=seed, evaluation=evaluation,
                                  failure_coverage=min(coverage, evaluation)))
    with (folder / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if with_curve:
        with (folder / "coverage_curve.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)


class StatisticsTests(unittest.TestCase):
    def test_wilcoxon_exact_matches_known_value(self):
        # Ranks 1..8 all positive: W- = 0, exact two-sided p = 2 / 2**8.
        statistic, p_value, effect, method = wilcoxon_signed_rank(np.arange(1, 9))
        self.assertEqual(statistic, 0)
        self.assertAlmostEqual(p_value, 2 / 2**8)
        self.assertEqual(effect, 1.0)
        self.assertEqual(method, "exact")

    def test_wilcoxon_is_symmetric_and_ignores_zeros(self):
        diffs = np.array([3.0, -1.0, 2.0, 0.0, 5.0, 4.0, -2.5, 6.0])
        _, p_pos, r_pos, _ = wilcoxon_signed_rank(diffs)
        _, p_neg, r_neg, _ = wilcoxon_signed_rank(-diffs)
        self.assertAlmostEqual(p_pos, p_neg)
        self.assertAlmostEqual(r_pos, -r_neg)
        self.assertEqual(wilcoxon_signed_rank([0, 0, 0])[1], 1.0)

    def test_wilcoxon_tied_small_and_large_samples(self):
        # Ties with n <= 13 use exact enumeration: all 5 positive -> p = 2 / 32.
        self.assertAlmostEqual(wilcoxon_signed_rank([1, 1, 2, 2, 2])[1], 2 / 32)
        self.assertEqual(wilcoxon_signed_rank([1, 1, 2, 2, 2])[3], "exact_ties")
        diffs = np.tile([1.0, 2.0, -1.0, 3.0], 10)
        _, p_value, _, method = wilcoxon_signed_rank(diffs)
        self.assertEqual(method, "normal")
        self.assertTrue(0 < p_value < 0.01)

    def test_holm_adjustment(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
        self.assertEqual(adjusted, [0.04, 0.09, 0.09, 0.5])

    def test_bootstrap_ci_contains_mean_and_is_reproducible(self):
        values = np.random.default_rng(5).normal(10, 2, 40)
        low, high = bootstrap_mean_ci(values, 2000, np.random.default_rng(1))
        self.assertLess(low, values.mean())
        self.assertGreater(high, values.mean())
        self.assertEqual((low, high), bootstrap_mean_ci(values, 2000, np.random.default_rng(1)))
        self.assertEqual(bootstrap_mean_ci([4, 4, 4], 200, np.random.default_rng(0)), (4.0, 4.0))


class AnalysisPipelineTests(unittest.TestCase):
    def test_censored_first_failure_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            write_results(Path(directory, "a"), 10, range(3), {"random": 2})
            runs = load_runs([Path(directory, "a")])
            censored = [run for run in runs if run.censored]
            self.assertEqual(len(censored), 1)
            self.assertEqual(censored[0].values["first_failure_evaluation"], 11)
            with self.assertRaises(ValueError):
                load_runs([Path(directory, "a"), Path(directory, "a", "summary.csv")])

    def test_full_analysis_over_two_budgets(self):
        methods = {"random": 3, "robustness_guided": 1, "diversity_aware": 6}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_results(root / "b10", 10, range(6), methods)
            write_results(root / "b20", 20, range(6), methods, with_curve=False)
            out = root / "analysis"
            stats, tests = run_analysis([root / "b10", root / "b20"], out, resamples=200)
            for name in ("summary_stats.csv", "paired_tests.csv", "results_table.md",
                         "metrics_ci.png", "coverage_curve_ci.png", "analysis_settings.json"):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(len(stats), 2 * 3 * 5)
            self.assertEqual(len(tests), 2 * 2 * 5)
            coverage = [t for t in tests if t.metric == "observed_failure_coverage"
                        and t.other == "random" and t.budget == 10][0]
            self.assertAlmostEqual(coverage.mean_difference, 3.0)
            self.assertTrue(all(t.p_holm >= t.p_value for t in tests))
            self.assertIn("Diversity-aware", (out / "results_table.md").read_text())
            settings = json.loads((out / "analysis_settings.json").read_text())
            self.assertEqual(settings["bootstrap_resamples"], 200)
            with self.assertRaises(FileExistsError):
                run_analysis([root / "b10"], out, resamples=200)

    def test_unknown_methods_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_results(root / "r", 10, range(4), {"diversity_aware": 5, "cma_archive": 4})
            _, tests = run_analysis([root / "r"], root / "out", resamples=200)
            self.assertEqual({t.other for t in tests}, {"cma_archive"})


if __name__ == "__main__":
    unittest.main()
