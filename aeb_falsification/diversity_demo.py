"""Compare all three falsification strategies."""

from .search import diversity_aware_search, random_search, robustness_guided_search


def main() -> None:
    budget = 100
    results = [
        random_search(budget=budget, seed=4),
        robustness_guided_search(budget=budget, seed=4),
        diversity_aware_search(budget=budget, seed=4),
    ]
    for result in results:
        best = result.best_record
        print(
            f"{result.method}: evaluations={result.evaluations}, "
            f"best_search_score={best.result.search_score:.3f}, "
            f"formal_robustness={best.result.metrics.overall_robustness:.3f}, "
            f"failure={best.result.failure_type}, "
            f"impact_speed={best.result.impact_speed:.3f} m/s, "
            f"archive_coverage={result.archive_coverage}"
        )
    diversity_result = results[-1]
    print("archive signatures:")
    for signature in diversity_result.archive_signatures:
        print(
            f"  - {signature.failure_type}, severity_bin={signature.severity_bin}, "
            f"friction={signature.friction_bin}, sensor={signature.sensor_bin}"
        )


if __name__ == "__main__":
    main()
