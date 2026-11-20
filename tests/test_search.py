import unittest

from aeb_falsification.search import random_search, robustness_guided_search


class SearchTests(unittest.TestCase):
    def test_random_search_respects_budget(self):
        result = random_search(budget=8, seed=3)
        self.assertEqual(result.evaluations, 8)
        self.assertEqual(result.method, "random")

    def test_robustness_search_respects_budget(self):
        result = robustness_guided_search(budget=8, seed=3)
        self.assertEqual(result.evaluations, 8)
        self.assertEqual(result.method, "robustness_guided")

    def test_search_is_reproducible(self):
        first = robustness_guided_search(budget=8, seed=3)
        second = robustness_guided_search(budget=8, seed=3)
        self.assertEqual(
            first.best_record.result.search_score,
            second.best_record.result.search_score,
        )


if __name__ == "__main__":
    unittest.main()
