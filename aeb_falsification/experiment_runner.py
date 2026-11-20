"""Equal-budget multi-seed experiment runner for the three search methods."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .archive import FailureArchive
from .search import diversity_aware_search, random_search, robustness_guided_search


@dataclass(frozen=True)
class ExperimentRow:
    method: str
    seed: int
    evaluations: int
    best_search_score: float
    best_formal_robustness: float
    best_failure_type: str
    best_impact_speed: float
    algorithm_archive_coverage: int
    observed_failure_coverage: int
    collisions: int
    violations: int
    first_failure_evaluation: int


def _run_method(method: str, budget: int, seed: int):
    if method == "random":
        return random_search(budget=budget, seed=seed)
    if method == "robustness_guided":
        return robustness_guided_search(budget=budget, seed=seed)
    if method == "diversity_aware":
        return diversity_aware_search(budget=budget, seed=seed)
    raise ValueError(f"unknown method: {method}")


def run_experiment(
    budget: int = 100,
    seeds: tuple[int, ...] = tuple(range(10)),
    output_dir: str | Path = "experiment_results",
) -> list[ExperimentRow]:
    """Run equal-budget experiments and write machine-readable results."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[ExperimentRow] = []
    for seed in seeds:
        for method in ("random", "robustness_guided", "diversity_aware"):
            result = _run_method(method, budget, seed)
            archive = FailureArchive()
            first_failure = -1
            collisions = 0
            violations = 0
            for record in result.records:
                archive.record(record.result, record.scenario)
                if record.result.failure_type is not None and first_failure < 0:
                    first_failure = record.evaluation
                if record.result.collision:
                    collisions += 1
                if record.result.failure_type is not None:
                    violations += 1

            best = result.best_record.result
            rows.append(
                ExperimentRow(
                    method=method,
                    seed=seed,
                    evaluations=result.evaluations,
                    best_search_score=best.search_score,
                    best_formal_robustness=best.metrics.overall_robustness,
                    best_failure_type=best.failure_type or "safe",
                    best_impact_speed=best.impact_speed,
                    algorithm_archive_coverage=result.archive_coverage,
                    observed_failure_coverage=archive.coverage,
                    collisions=collisions,
                    violations=violations,
                    first_failure_evaluation=first_failure,
                )
            )

    _write_csv(rows, output_path / "summary.csv")
    _write_summary_plot(rows, output_path / "summary.png")
    return rows


def _write_csv(rows: list[ExperimentRow], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_summary_plot(rows: list[ExperimentRow], path: Path) -> None:
    methods = ("random", "robustness_guided", "diversity_aware")
    labels = ("Random", "Robustness-guided", "Diversity-aware")
    coverage = [
        [row.observed_failure_coverage for row in rows if row.method == method]
        for method in methods
    ]
    scores = [
        [row.best_search_score for row in rows if row.method == method]
        for method in methods
    ]
    x = np.arange(len(methods))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x, [np.mean(values) for values in coverage], width, color="tab:blue")
    axes[0].set_title("Observed failure coverage")
    axes[0].set_ylabel("Unique archive cells")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot(scores, tick_labels=labels, showmeans=True)
    axes[1].set_title("Best search score across seeds")
    axes[1].set_ylabel("Lower is more severe")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    args = parser.parse_args()
    rows = run_experiment(
        budget=args.budget,
        seeds=tuple(range(args.seeds)),
        output_dir=args.output_dir,
    )
    print(f"wrote {len(rows)} runs to {args.output_dir / 'summary.csv'}")
    print(f"wrote plot to {args.output_dir / 'summary.png'}")
    for method in ("random", "robustness_guided", "diversity_aware"):
        method_rows = [row for row in rows if row.method == method]
        print(
            f"{method}: mean_coverage="
            f"{np.mean([row.observed_failure_coverage for row in method_rows]):.2f}, "
            f"mean_best_score="
            f"{np.mean([row.best_search_score for row in method_rows]):.3f}"
        )


if __name__ == "__main__":
    main()
