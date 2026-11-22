"""Equal-budget multi-seed experiment runner for the three search methods."""

from __future__ import annotations

import argparse
import csv
import json
from importlib.metadata import version
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .archive import FailureArchive
from .models import SimulationConfig
from .search import diversity_aware_search, random_search, robustness_guided_search
from .search.common import BOUNDS, PARAMETER_NAMES

DEFAULT_OUTPUT_DIR = Path("experiment_results/latest")


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
    worst_impact_speed_kmh: float
    collision_rate: float
    violation_rate: float


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
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[ExperimentRow]:
    """Run equal-budget experiments and write machine-readable results."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if budget <= 0:
        raise ValueError("budget must be positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[ExperimentRow] = []
    failures: list[dict] = []
    curves: list[dict] = []
    for seed in seeds:
        for method in ("random", "robustness_guided", "diversity_aware"):
            result = _run_method(method, budget, seed)
            archive = FailureArchive()
            first_failure = -1
            collisions = 0
            violations = 0
            for record in result.records:
                if record.result.failure_type is not None:
                    archive.record(record.result, record.scenario)
                if record.result.failure_type is not None and first_failure < 0:
                    first_failure = record.evaluation + 1
                if record.result.collision:
                    collisions += 1
                if record.result.failure_type is not None:
                    violations += 1
                curves.append({
                    "method": method, "seed": seed, "evaluation": record.evaluation + 1,
                    "failure_coverage": archive.coverage,
                })
            for entry in archive.entries():
                failures.append({
                    "method": method,
                    "seed": seed,
                    "category": asdict(entry.signature),
                    "scenario": asdict(entry.scenario),
                    "outcome": outcome(entry.result),
                })

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
                    algorithm_archive_coverage=sum(
                        signature.failure_type != "safe" for signature in result.archive_signatures
                    ),
                    observed_failure_coverage=archive.coverage,
                    collisions=collisions,
                    violations=violations,
                    first_failure_evaluation=first_failure,
                    worst_impact_speed_kmh=max(
                        (record.result.impact_speed * 3.6 for record in result.records if record.result.collision),
                        default=0.0,
                    ),
                    collision_rate=collisions / result.evaluations,
                    violation_rate=violations / result.evaluations,
                )
            )

    _write_csv(rows, output_path / "summary.csv")
    _write_summary_plot(rows, output_path / "summary.png")
    _write_curve(curves, budget, output_path)
    simulation = asdict(SimulationConfig())
    _write_json({"simulation": simulation, "failures": failures}, output_path / "failures.json")
    _write_json({
        "budget_per_method_per_seed": budget,
        "seeds": list(seeds),
        "simulation": simulation,
        "bounds": dict(zip(PARAMETER_NAMES, BOUNDS.tolist())),
        "versions": {name: version(name) for name in ("cma", "numpy", "matplotlib")},
    }, output_path / "settings.json")
    return rows


def outcome(result) -> dict:
    """Values a replay must reproduce."""
    return {
        "failure_type": result.failure_type,
        "impact_speed": result.impact_speed,
        "min_distance": result.min_distance,
        "min_ttc": result.min_ttc,
        "search_score": result.search_score,
        "violated_properties": list(result.violated_properties),
    }


def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n")


def _write_curve(curves: list[dict], budget: int, output_path: Path) -> None:
    """Write mean failure coverage against simulations used (band = 1 SD)."""
    with (output_path / "coverage_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for method, label, color in (
        ("random", "Random", "#377eb8"),
        ("robustness_guided", "CMA-ES", "#e68613"),
        ("diversity_aware", "Diversity-aware", "#36945d"),
    ):
        values = np.array([
            row["failure_coverage"] for row in curves if row["method"] == method
        ]).reshape(-1, budget)
        mean, spread = values.mean(axis=0), values.std(axis=0)
        x = np.arange(1, budget + 1)
        axis.plot(x, mean, label=label, color=color)
        axis.fill_between(x, mean - spread, mean + spread, color=color, alpha=.15)
    axis.set_xlabel("Simulations used")
    axis.set_ylabel("Distinct failure categories (mean, 1 SD band)")
    axis.grid(alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path / "coverage_curve.png", dpi=180)
    plt.close(fig)


def _write_csv(rows: list[ExperimentRow], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_summary_plot(rows: list[ExperimentRow], path: Path) -> None:
    methods = ("random", "robustness_guided", "diversity_aware")
    labels = ("Random", "CMA-ES", "Diversity-aware")
    coverage = [
        [row.observed_failure_coverage for row in rows if row.method == method]
        for method in methods
    ]
    impact_speeds = [
        [row.worst_impact_speed_kmh for row in rows if row.method == method]
        for method in methods
    ]
    colors = ("#377eb8", "#e68613", "#36945d")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, groups, title, ylabel in (
        (axes[0], coverage, "Variety of failures discovered", "Distinct failure categories (higher = broader)"),
        (axes[1], impact_speeds, "Worst collision found in each run", "Ego impact speed, km/h (higher = more severe)"),
    ):
        for index, (values, color) in enumerate(zip(groups, colors)):
            mean = float(np.mean(values))
            axis.bar(index, mean, width=.6, color=color, alpha=.65)
            axis.scatter(index + np.linspace(-.16, .16, len(values)), values,
                         color=color, edgecolors="white", linewidths=.6, zorder=3)
            axis.annotate(f"{mean:.1f}", (index, max(values)), xytext=(0, 9),
                          textcoords="offset points", ha="center", weight="bold")
        axis.set_title(title, pad=14)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(3), labels)
        axis.set_ylim(0, max(max(values) for values in groups) * 1.2 + 1)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    budgets = sorted({row.evaluations for row in rows})
    counts = [sum(row.method == method for row in rows) for method in methods]
    fig.suptitle(f"AEB testing comparison | {budgets[0]} simulations per run | {counts[0]} runs per method", fontsize=14)
    fig.text(.5, .025, "Bars and numbers = averages; each dot = one run. Safe categories excluded.\n"
             "No collision in a run is recorded as 0 km/h. Categories describe outcomes and conditions, not proven causes.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
            f"mean_worst_impact_kmh="
            f"{np.mean([row.worst_impact_speed_kmh for row in method_rows]):.2f}"
        )


if __name__ == "__main__":
    main()
