import csv
import tempfile
import unittest
from pathlib import Path

from aeb_falsification.experiment_runner import run_experiment


class ExperimentRunnerTests(unittest.TestCase):
    def test_equal_budget_multi_seed_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = run_experiment(budget=5, seeds=(0, 1), output_dir=directory)
            self.assertEqual(len(rows), 6)
            self.assertTrue(Path(directory, "summary.csv").exists())
            self.assertTrue(Path(directory, "summary.png").exists())
            with Path(directory, "summary.csv").open() as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 6)
            self.assertEqual({row["evaluations"] for row in csv_rows}, {"5"})


if __name__ == "__main__":
    unittest.main()


class MetricDefinitionTests(unittest.TestCase):
    def test_safe_results_do_not_count_as_failure_coverage(self):
        from unittest.mock import patch
        from aeb_falsification import Scenario, simulate
        from aeb_falsification.search.common import SearchRecord, SearchResult
        safe_scenario = Scenario(8, 60, 2, 1, 0, 0, 1)
        crash_scenario = Scenario(20, 8, 0, .35, .5, 0, 1)
        safe = simulate(safe_scenario)
        crash = simulate(crash_scenario)
        records = [SearchRecord(0, safe_scenario, safe), SearchRecord(1, crash_scenario, crash)]
        result = SearchResult('test', records, records[1])
        with tempfile.TemporaryDirectory() as directory, patch(
            'aeb_falsification.experiment_runner._run_method', return_value=result
        ):
            rows = run_experiment(budget=2, seeds=(0,), output_dir=directory)
        for row in rows:
            self.assertEqual(row.observed_failure_coverage, 1)
            self.assertEqual(row.first_failure_evaluation, 2)
            self.assertAlmostEqual(row.worst_impact_speed_kmh, crash.impact_speed * 3.6)
            self.assertEqual(row.collision_rate, .5)
            self.assertEqual(row.violation_rate, .5)

    def test_no_failure_run_has_zero_coverage_and_impact(self):
        from unittest.mock import patch
        from aeb_falsification import Scenario, simulate
        from aeb_falsification.search.common import SearchRecord, SearchResult
        scenario = Scenario(8, 60, 2, 1, 0, 0, 1)
        record = SearchRecord(0, scenario, simulate(scenario))
        result = SearchResult('test', [record], record)
        with tempfile.TemporaryDirectory() as directory, patch(
            'aeb_falsification.experiment_runner._run_method', return_value=result
        ):
            rows = run_experiment(budget=1, seeds=(0,), output_dir=directory)
        for row in rows:
            self.assertEqual(row.observed_failure_coverage, 0)
            self.assertEqual(row.worst_impact_speed_kmh, 0)
            self.assertEqual(row.first_failure_evaluation, -1)
