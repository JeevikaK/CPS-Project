"""Compare the two first search baselines."""

from .search import random_search, robustness_guided_search


def main() -> None:
    budget = 100
    random_result = random_search(budget=budget, seed=4)
    guided_result = robustness_guided_search(budget=budget, seed=4)
    for result in (random_result, guided_result):
        best = result.best_record
        print(
            f"{result.method}: evaluations={result.evaluations}, "
            f"best_search_score={best.result.search_score:.3f}, "
            f"formal_robustness={best.result.metrics.overall_robustness:.3f}, "
            f"failure={best.result.failure_type}, "
            f"min_distance={best.result.min_distance:.3f} m, "
            f"impact_speed={best.result.impact_speed:.3f} m/s"
        )


if __name__ == "__main__":
    main()
