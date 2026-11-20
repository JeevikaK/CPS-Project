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
