"""Archive ablation: the same CMA-ES with and without the failure archive.

Run everything with one command:

    python -m aeb_falsification.ablation

Every CMA-ES arm uses the identical optimizer (``search.cma_es.cma_search``), budget,
seeds, and noise process; arms differ only in the switches below. Uniform
random search is included as a reference. Paired statistics compare each arm
with plain CMA-ES (``cma``) on the same seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .archive import FailureArchive
from .models import SimulationConfig
from .search import cma_search, diversity_aware_search, random_search
from .search.common import BOUNDS, DEFAULT_NOISE_MODE, NOISE_MODES, PARAMETER_NAMES, noise_seed
from .simulator import simulate

DEFAULT_OUTPUT_DIR = Path("experiment_results/latest_ablation")
# Settings were chosen on seeds 5000-5029; these seeds were not used for tuning.
DEFAULT_SEEDS = tuple(range(6000, 6030))
NOVELTY_BONUS = 30.0
REFERENCE_ARM = "cma"

# name -> (label, cma_search options, or None for a non-CMA reference method)
ARMS: dict[str, tuple[str, dict | None]] = {
    "random": ("Random (reference)", None),
    "diversity_aware": ("Archive mutation search (reference)", None),
    "cma_no_restart": ("CMA-ES, no stall restarts", {"stall_generations": None}),
    "cma": ("CMA-ES (no archive)", {}),
    "cma_novelty": ("+ archive novelty fitness", {"novelty_bonus": NOVELTY_BONUS}),
    "cma_archive_restarts": ("+ archive restarts", {"archive_restarts": True}),
    "archive_cma": ("+ both (archive-guided CMA-ES)",
                    {"novelty_bonus": NOVELTY_BONUS, "archive_restarts": True}),
}
SHORT_LABELS = {
    "random": "Random\n(ref.)", "diversity_aware": "Mutation\nsearch (ref.)",
    "cma_no_restart": "CMA-ES\nno restarts", "cma": "CMA-ES\n(baseline)",
    "cma_novelty": "+ novelty\nfitness", "cma_archive_restarts": "+ archive\nrestarts",
    "archive_cma": "+ both\n(archive CMA)",
}
METRICS = ("failure_coverage", "worst_impact_kmh", "collisions", "first_failure_evaluation")


@dataclass(frozen=True)
class AblationRow:
    arm: str
    noise: str
    seed: int
    evaluations: int
    failure_coverage: int
    worst_impact_kmh: float
    collisions: int
    violations: int
    first_failure_evaluation: int
    restarts: int
    archive_seeded_restarts: int
    reseed_failure_rate: float


def run_arm(arm: str, budget: int, seed: int, noise: str = DEFAULT_NOISE_MODE):
    options = ARMS[arm][1]
    if arm == "random":
        return random_search(budget, seed, noise=noise)
    if arm == "diversity_aware":
        return diversity_aware_search(budget, seed, noise=noise)
    return cma_search(budget, seed, noise=noise, method=arm, **options)


def reseed_failure_rate(archive: FailureArchive, reseeds: int, salt: int) -> float:
    """Fraction of archived failures that still fail with fresh noise seeds.

    Each category's stored worst scenario is re-simulated ``reseeds`` times with
    unrelated noise seeds. Low values mean failures depend on one noise draw.
    """
    trials = [
        simulate(replace(entry.scenario, seed=noise_seed(salt, 10_000 + index * reseeds + k))).failure_type
        is not None
        for index, entry in enumerate(archive.entries())
        for k in range(reseeds)
    ]
    return float(np.mean(trials)) if trials else float("nan")


def summarize(arm: str, noise: str, seed: int, result, reseeds: int) -> tuple[AblationRow, list[int]]:
    archive = FailureArchive()
    curve, first_failure, collisions, violations = [], -1, 0, 0
    for record in result.records:
        archive.record(record.result, record.scenario)
        curve.append(archive.coverage)
        if record.result.failure_type is not None:
            violations += 1
            if first_failure < 0:
                first_failure = record.evaluation + 1
        collisions += record.result.collision
    run = result.optimizer
    row = AblationRow(
        arm=arm,
        noise=noise,
        seed=seed,
        evaluations=result.evaluations,
        failure_coverage=archive.coverage,
        worst_impact_kmh=max((r.result.impact_speed * 3.6 for r in result.records if r.result.collision),
                             default=0.0),
        collisions=collisions,
        violations=violations,
        first_failure_evaluation=first_failure,
        restarts=run.restarts if run else 0,
        archive_seeded_restarts=run.archive_seeded_restarts if run else 0,
        reseed_failure_rate=reseed_failure_rate(archive, reseeds, seed) if reseeds else float("nan"),
    )
    return row, curve


def paired_test(differences: np.ndarray, rng: np.random.Generator, resamples: int = 20_000):
    """Mean paired difference, 95% bootstrap CI, two-sided sign-flip p-value."""
    differences = np.asarray(differences, dtype=float)
    mean = float(differences.mean())
    boot = rng.choice(differences, (resamples, differences.size)).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    if not np.any(differences):
        return mean, float(low), float(high), 1.0
    signs = rng.choice((-1.0, 1.0), (resamples, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    p = (1 + np.sum(null >= abs(mean) - 1e-12)) / (resamples + 1)
    return mean, float(low), float(high), float(p)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def compare(rows: list[AblationRow], noise: str) -> list[dict]:
    """Paired comparison of each arm with plain CMA-ES, Holm-corrected per metric."""
    rng = np.random.default_rng(0)
    table = {(row.arm, row.seed): row for row in rows if row.noise == noise}
    seeds = sorted({seed for arm, seed in table if arm == REFERENCE_ARM})
    comparisons = []
    for metric in METRICS:
        block = []
        for arm in ARMS:
            if arm == REFERENCE_ARM or (arm, seeds[0]) not in table:
                continue
            values = [getattr(table[arm, s], metric) for s in seeds]
            reference = [getattr(table[REFERENCE_ARM, s], metric) for s in seeds]
            mean, low, high, p = paired_test(np.subtract(values, reference), rng)
            block.append({"noise": noise, "metric": metric, "arm": arm, "versus": REFERENCE_ARM,
                          "arm_mean": float(np.mean(values)), "versus_mean": float(np.mean(reference)),
                          "mean_difference": mean, "ci95_low": low, "ci95_high": high, "p_value": p})
        for item, adjusted in zip(block, holm([item["p_value"] for item in block])):
            item["p_holm"] = adjusted
        comparisons += block
    return comparisons


def run_ablation(
    budget: int = 100,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    noise_modes: tuple[str, ...] = (DEFAULT_NOISE_MODE,),
    reseeds: int = 5,
    arms: tuple[str, ...] = tuple(ARMS),
) -> tuple[list[AblationRow], list[dict]]:
    if budget <= 0 or not seeds:
        raise ValueError("budget must be positive and at least one seed is required")
    if REFERENCE_ARM not in arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"arms must be drawn from {tuple(ARMS)} and include {REFERENCE_ARM!r}")
    if any(mode not in NOISE_MODES for mode in noise_modes):
        raise ValueError(f"noise modes must be drawn from {NOISE_MODES}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for noise in noise_modes:
        for seed in seeds:
            for arm in arms:
                row, curve = summarize(arm, noise, seed, run_arm(arm, budget, seed, noise), reseeds)
                rows.append(row)
                curves.setdefault((arm, noise), []).append(curve)
    comparisons = [item for noise in noise_modes for item in compare(rows, noise)]

    _write_csv([asdict(row) for row in rows], output / "summary.csv")
    _write_csv(comparisons, output / "comparison.csv")
    _write_csv([{"arm": arm, "noise": noise, "seed": seed, "evaluation": index + 1, "failure_coverage": value}
                for (arm, noise), runs in curves.items() for seed, curve in zip(seeds, runs)
                for index, value in enumerate(curve)], output / "coverage_curve.csv")
    _write_plot(curves, rows, noise_modes, budget, len(seeds), output / "ablation.png")
    (output / "results.md").write_text(_markdown(rows, comparisons, noise_modes, budget, seeds), encoding="utf-8")
    (output / "settings.json").write_text(json.dumps({
        "budget_per_arm_per_seed": budget,
        "seeds": list(seeds),
        "noise_modes": list(noise_modes),
        "reseeds_per_failure": reseeds,
        "arms": {arm: {"label": ARMS[arm][0], "options": ARMS[arm][1]} for arm in arms},
        "cma": {"sigma0": 0.2, "restart_sigma": 0.15, "stall_generations": 4, "stall_statistic": "generation median",
                "stall_tolerance_relative": 1e-3, "min_step": 1e-3, "restart_exploration": 0.3,
                "coordinates": "normalized [0, 1]"},
        "simulation": asdict(SimulationConfig()),
        "bounds": dict(zip(PARAMETER_NAMES, BOUNDS.tolist())),
        "versions": {name: version(name) for name in ("cma", "numpy", "matplotlib")},
        "statistics": "paired vs cma on same seeds; mean difference, 95% bootstrap CI, "
                      "two-sided sign-flip permutation p (20000 resamples), Holm per metric and noise mode",
    }, indent=1) + "\n", encoding="utf-8")
    return rows, comparisons


def _write_csv(items: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)


def _arm_means(rows, arm, noise):
    chosen = [row for row in rows if row.arm == arm and row.noise == noise]
    return {field: float(np.mean([getattr(row, field) for row in chosen])) for field in
            ("failure_coverage", "worst_impact_kmh", "collisions", "restarts",
             "archive_seeded_restarts", "reseed_failure_rate")}, \
        float(np.std([row.failure_coverage for row in chosen]))


def _markdown(rows, comparisons, noise_modes, budget, seeds) -> str:
    lines = [f"# Archive ablation ({budget} simulations per arm, {len(seeds)} seeds: "
             f"{seeds[0]}–{seeds[-1]})", ""]
    for noise in noise_modes:
        lines += [f"## Noise seed: `{noise}`", "",
                  "| Arm | Failure categories (mean ± SD) | Worst impact (km/h) | Collisions / run "
                  "| Restarts | Archive-seeded | Failures that survive reseeding |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            if not any(row.arm == arm and row.noise == noise for row in rows):
                continue
            mean, sd = _arm_means(rows, arm, noise)
            lines.append(f"| {ARMS[arm][0]} | {mean['failure_coverage']:.1f} ± {sd:.1f} | "
                         f"{mean['worst_impact_kmh']:.1f} | {mean['collisions']:.1f} | "
                         f"{mean['restarts']:.1f} | {mean['archive_seeded_restarts']:.1f} | "
                         f"{mean['reseed_failure_rate']:.0%} |")
        lines += ["", "Paired difference against plain CMA-ES (95% bootstrap CI; Holm-corrected p):", "",
                  "| Arm | Δ categories | Δ worst impact (km/h) |", "|---|---|---|"]
        for arm in ARMS:
            cells = {c["metric"]: c for c in comparisons
                     if c["arm"] == arm and c["noise"] == noise}
            if not cells:
                continue
            fmt = lambda c: (f"{c['mean_difference']:+.1f} [{c['ci95_low']:+.1f}, "
                             f"{c['ci95_high']:+.1f}], p={c['p_holm']:.2g}")
            lines.append(f"| {ARMS[arm][0]} | {fmt(cells['failure_coverage'])} | "
                         f"{fmt(cells['worst_impact_kmh'])} |")
        lines.append("")
    lines += ["Categories are failure-only archive cells (outcome × severity bin × friction × sensor "
              "region). Impact is ego speed at contact; 0 when a run has no collision. "
              "Reseeding re-simulates each archived worst failure with 5 fresh noise seeds.", ""]
    return "\n".join(lines)


def _write_plot(curves, rows, noise_modes, budget, runs, path: Path) -> None:
    colors = {"random": "#8c8c8c", "diversity_aware": "#b5487a", "cma_no_restart": "#c9a227", "cma": "#e68613",
              "cma_novelty": "#7b68b5", "cma_archive_restarts": "#3a8fb7", "archive_cma": "#36945d"}
    noise = noise_modes[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(1, budget + 1)
    present = [arm for arm in ARMS if (arm, noise) in curves]
    for arm in present:
        values = np.asarray(curves[arm, noise], dtype=float)
        mean = values.mean(axis=0)
        left.plot(x, mean, color=colors[arm], label=ARMS[arm][0], lw=2 if arm in ("cma", "archive_cma") else 1.3)
        if arm in ("cma", "archive_cma"):
            left.fill_between(x, mean - values.std(axis=0), mean + values.std(axis=0),
                              color=colors[arm], alpha=.12)
    left.set_xlabel("Simulations used")
    left.set_ylabel("Distinct failure categories\n(mean; 1 SD band for baseline and archive CMA-ES)")
    left.set_title("Failure coverage against budget")
    left.legend(fontsize=8, frameon=False)
    for index, arm in enumerate(present):
        values = [row.failure_coverage for row in rows if row.arm == arm and row.noise == noise]
        right.bar(index, np.mean(values), color=colors[arm], alpha=.65, width=.6)
        right.scatter(index + np.linspace(-.18, .18, len(values)), values, s=10, color=colors[arm], zorder=3)
        right.annotate(f"{np.mean(values):.1f}", (index, max(values)), xytext=(0, 6),
                       textcoords="offset points", ha="center", weight="bold")
    right.set_xticks(range(len(present)), [SHORT_LABELS[arm] for arm in present], fontsize=8)
    right.set_ylabel("Distinct failure categories at end of run")
    right.set_title("Final coverage per run (bar = mean, dot = run)")
    for axis in (left, right):
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Archive ablation | {budget} simulations per run | {runs} runs per arm | noise: {noise}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("FIRST", "COUNT"),
                        default=(DEFAULT_SEEDS[0], len(DEFAULT_SEEDS)))
    parser.add_argument("--noise", nargs="+", choices=NOISE_MODES, default=[DEFAULT_NOISE_MODE],
                        help="noise-seed modes to run (several give the noise-seed study)")
    parser.add_argument("--reseeds", type=int, default=5,
                        help="fresh noise seeds per archived failure for the reseeding check (0 = skip)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    first, count = args.seeds
    rows, _ = run_ablation(args.budget, tuple(range(first, first + count)), args.output_dir,
                           tuple(args.noise), args.reseeds)
    print((args.output_dir / "results.md").read_text(encoding="utf-8").encode("ascii", "replace").decode())
    print(f"wrote {len(rows)} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
