"""Statistics, confidence intervals and final plots for experiment results.

Run:
    python -m aeb_falsification.analysis <result folder> [more folders] --output-dir <new folder>

Example:
    python -m aeb_falsification.analysis experiment_results/archive_guided \\
        --output-dir experiment_results/latest/analysis

How it works: reads summary.csv from each folder, computes mean and 95% CI per
method, compares diversity_aware to the other methods seed by seed (Wilcoxon
test, Holm-corrected), then writes tables (.csv, .md) and plots (.png).

Overwrite rule: if the output folder already has files, the script stops.
Pick a new folder, or add --overwrite to replace them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("experiment_results/latest/analysis")
DEFAULT_REFERENCE = "diversity_aware"
CONFIDENCE = 0.95

# (column, label). For simulations to first failure, lower means found sooner.
METRICS: tuple[tuple[str, str], ...] = (
    ("observed_failure_coverage", "Distinct failure categories"),
    ("worst_impact_speed_kmh", "Worst impact speed (km/h)"),
    ("violations", "Violating simulations"),
    ("collisions", "Collisions"),
    ("first_failure_evaluation", "Simulations to first failure"),
)

# Fixed method order and styling; unknown methods are appended after these.
METHOD_STYLES: dict[str, tuple[str, str, str]] = {
    "random": ("Random", "#377eb8", "o"),
    "robustness_guided": ("CMA-ES", "#e68613", "s"),
    "diversity_aware": ("Diversity-aware", "#36945d", "D"),
}
EXTRA_STYLES = (("#6a4ce2", "^"), ("#ee7fac", "v"), ("#008a17", "P"))


@dataclass(frozen=True)
class Run:
    method: str
    seed: int
    budget: int
    values: dict[str, float]
    censored: bool  # no failure found within the budget


@dataclass(frozen=True)
class SummaryStat:
    budget: int
    method: str
    metric: str
    n: int
    mean: float
    sd: float
    median: float
    ci_low: float
    ci_high: float
    censored_runs: int


@dataclass(frozen=True)
class PairedTest:
    budget: int
    metric: str
    reference: str
    other: str
    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    wilcoxon_statistic: float
    p_value: float
    p_holm: float
    rank_biserial: float
    test_method: str


# --------------------------------------------------------------------------- loading


def load_runs(paths: list[str | Path]) -> list[Run]:
    """Load runs from result folders or ``summary.csv`` files.

    ``first_failure_evaluation`` of -1 (no failure found) is treated as censored
    and replaced by ``budget + 1``, a conservative lower bound.
    """
    runs: list[Run] = []
    seen: set[tuple[int, str, int]] = set()
    for path in paths:
        csv_path = _summary_path(Path(path))
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                budget = int(row["evaluations"])
                key = (budget, row["method"], int(row["seed"]))
                if key in seen:
                    raise ValueError(f"duplicate run {key} in {csv_path}")
                seen.add(key)
                values = {metric: float(row[metric]) for metric, _ in METRICS}
                censored = values["first_failure_evaluation"] < 0
                if censored:
                    values["first_failure_evaluation"] = float(budget + 1)
                runs.append(Run(row["method"], key[2], budget, values, censored))
    if not runs:
        raise ValueError("no runs found")
    return runs


def _summary_path(path: Path) -> Path:
    candidate = path / "summary.csv" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"no summary.csv at {path}")
    return candidate


def method_order(methods) -> list[str]:
    known = [method for method in METHOD_STYLES if method in methods]
    return known + sorted(set(methods) - set(METHOD_STYLES))


def method_style(method: str, order: list[str]) -> tuple[str, str, str]:
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    extras = [m for m in order if m not in METHOD_STYLES]
    color, marker = EXTRA_STYLES[extras.index(method) % len(EXTRA_STYLES)]
    return method.replace("_", " "), color, marker


# --------------------------------------------------------------------------- statistics


def bootstrap_mean_ci(
    values, resamples: int, rng: np.random.Generator, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return math.nan, math.nan
    if data.size == 1 or np.all(data == data[0]):
        return float(data[0]), float(data[0])
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return float(low), float(high)


def wilcoxon_signed_rank(differences) -> tuple[float, float, float, str]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are dropped (Wilcoxon's method). Uses the exact null
    distribution when there are no ties and at most 50 pairs, an exact
    enumeration of all sign flips when there are ties and at most 13 pairs,
    otherwise the normal approximation with tie correction (no continuity
    correction). This matches ``scipy.stats.wilcoxon`` defaults, except that
    the pair counts are taken after dropping zeros, so a few cases SciPy
    approximates are computed exactly here. Returns
    ``(W, p_value, rank_biserial, method)`` where ``W = min(W+, W-)`` and the
    rank-biserial correlation is ``(W+ - W-) / (W+ + W-)``.
    """
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 1.0, 0.0, "all_zero"
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = n * (n + 1) / 2
    rank_biserial = (w_plus - w_minus) / total
    _, tie_counts = np.unique(np.abs(d), return_counts=True)
    has_ties = bool(np.any(tie_counts > 1))
    if n <= 50 and not has_ties:
        counts = _exact_signed_rank_counts(n)
        cdf = np.cumsum(counts) / counts.sum()
        w = int(round(w_plus))
        lower = cdf[w]
        upper = 1 - (cdf[w - 1] if w > 0 else 0.0)
        p_value = min(1.0, 2 * min(lower, upper))
        method = "exact"
    elif n <= 13:
        signs = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
        null = (signs * ranks).sum(axis=1)
        tolerance = 1e-9
        lower = float(np.mean(null <= w_plus + tolerance))
        upper = float(np.mean(null >= w_plus - tolerance))
        p_value = min(1.0, 2 * min(lower, upper))
        method = "exact_ties"
    else:
        variance = n * (n + 1) * (2 * n + 1) / 24
        variance -= float(np.sum(tie_counts**3 - tie_counts)) / 48
        if variance <= 0:
            return min(w_plus, w_minus), 1.0, rank_biserial, "normal"
        z = (w_plus - total / 2) / math.sqrt(variance)
        p_value = min(1.0, math.erfc(abs(z) / math.sqrt(2)))
        method = "normal"
    return min(w_plus, w_minus), float(p_value), float(rank_biserial), method


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def _exact_signed_rank_counts(n: int) -> np.ndarray:
    """Number of sign assignments giving each W+ value for ranks 1..n."""
    counts = np.zeros(n * (n + 1) // 2 + 1, dtype=float)
    counts[0] = 1
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy()
    return counts


def holm_adjust(p_values) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for position, index in enumerate(np.argsort(p, kind="mergesort")):
        running = max(running, min(1.0, (m - position) * p[index]))
        adjusted[index] = running
    return adjusted.tolist()


def summarize(runs: list[Run], resamples: int, seed: int) -> list[SummaryStat]:
    rng = np.random.default_rng(seed)
    stats: list[SummaryStat] = []
    for budget in sorted({run.budget for run in runs}):
        budget_runs = [run for run in runs if run.budget == budget]
        for method in method_order({run.method for run in budget_runs}):
            group = [run for run in budget_runs if run.method == method]
            for metric, _ in METRICS:
                values = np.array([run.values[metric] for run in group])
                low, high = bootstrap_mean_ci(values, resamples, rng)
                stats.append(SummaryStat(
                    budget=budget, method=method, metric=metric, n=values.size,
                    mean=float(values.mean()),
                    sd=float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    median=float(np.median(values)), ci_low=low, ci_high=high,
                    censored_runs=sum(run.censored for run in group)
                    if metric == "first_failure_evaluation" else 0,
                ))
    return stats


def paired_tests(
    runs: list[Run], reference: str, resamples: int, seed: int
) -> list[PairedTest]:
    """Reference minus other, paired by seed. Holm correction over all tests."""
    rng = np.random.default_rng(seed + 1)
    raw: list[dict] = []
    for budget in sorted({run.budget for run in runs}):
        by_method: dict[str, dict[int, Run]] = {}
        for run in runs:
            if run.budget == budget:
                by_method.setdefault(run.method, {})[run.seed] = run
        if reference not in by_method:
            continue
        for other in method_order(by_method):
            if other == reference:
                continue
            seeds = sorted(set(by_method[reference]) & set(by_method[other]))
            if len(seeds) < 2:
                continue
            for metric, _ in METRICS:
                diffs = np.array([
                    by_method[reference][s].values[metric] - by_method[other][s].values[metric]
                    for s in seeds
                ])
                low, high = bootstrap_mean_ci(diffs, resamples, rng)
                statistic, p_value, effect, test_method = wilcoxon_signed_rank(diffs)
                raw.append(dict(
                    budget=budget, metric=metric, reference=reference, other=other,
                    n_pairs=len(seeds), mean_difference=float(diffs.mean()),
                    ci_low=low, ci_high=high, wilcoxon_statistic=statistic,
                    p_value=p_value, rank_biserial=effect, test_method=test_method,
                ))
    adjusted = holm_adjust([row["p_value"] for row in raw]) if raw else []
    return [PairedTest(p_holm=p, **row) for row, p in zip(raw, adjusted)]


# --------------------------------------------------------------------------- outputs


def write_tables(stats: list[SummaryStat], tests: list[PairedTest], output: Path) -> None:
    for name, rows in (("summary_stats.csv", stats), ("paired_tests.csv", tests)):
        with (output / name).open("w", newline="") as stream:
            fields = list(asdict(rows[0])) if rows else []
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    (output / "results_table.md").write_text(markdown_tables(stats, tests))


def markdown_tables(stats: list[SummaryStat], tests: list[PairedTest]) -> str:
    labels = dict(METRICS)
    lines = [
        "# Results",
        "",
        f"Mean [{CONFIDENCE:.0%} bootstrap CI] over seeds. Runs with no failure count",
        "as budget + 1 simulations to first failure (shown as censored).",
        "",
    ]
    for budget in sorted({s.budget for s in stats}):
        budget_stats = [s for s in stats if s.budget == budget]
        methods = method_order({s.method for s in budget_stats})
        n = max(s.n for s in budget_stats)
        lines += [f"## {budget} simulations per run (n = {n} seeds)", ""]
        lines.append("| Method | " + " | ".join(labels[m] for m, _ in METRICS) + " |")
        lines.append("|---|" + "---:|" * len(METRICS))
        for method in methods:
            cells = []
            for metric, _ in METRICS:
                s = next(x for x in budget_stats if x.method == method and x.metric == metric)
                cell = f"{s.mean:.1f} [{s.ci_low:.1f}, {s.ci_high:.1f}]"
                if s.censored_runs:
                    cell += f" ({s.censored_runs} censored)"
                cells.append(cell)
            label = method_style(method, methods)[0]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    if tests:
        reference = tests[0].reference
        ref_label = METHOD_STYLES.get(reference, (reference,))[0]
        lines += [
            f"## Paired comparisons: {ref_label} minus each other method",
            "",
            "Wilcoxon signed-rank, two-sided, paired by seed. Holm correction across",
            f"all {len(tests)} tests. Effect size = matched-pairs rank-biserial r.",
            "",
            "| Budget | Metric | vs | Mean difference [95% CI] | p | p (Holm) | r |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        order = method_order({t.other for t in tests})
        for t in tests:
            lines.append(
                f"| {t.budget} | {labels[t.metric]} | {method_style(t.other, order)[0]} | "
                f"{t.mean_difference:+.2f} [{t.ci_low:+.2f}, {t.ci_high:+.2f}] | "
                f"{t.p_value:.2g} | {t.p_holm:.2g} | {t.rank_biserial:+.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_metrics(runs: list[Run], stats: list[SummaryStat], path: Path) -> None:
    budgets = sorted({run.budget for run in runs})
    methods = method_order({run.method for run in runs})
    columns = 3
    rows = math.ceil(len(METRICS) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.9 * rows), squeeze=False)
    width = 0.8 / len(methods)
    for axis, (metric, label) in zip(axes.flat, METRICS):
        for index, method in enumerate(methods):
            name, color, marker = method_style(method, methods)
            for b_index, budget in enumerate(budgets):
                x = b_index + (index - (len(methods) - 1) / 2) * width
                values = [r.values[metric] for r in runs if r.method == method and r.budget == budget]
                if not values:
                    continue
                s = next(st for st in stats if st.method == method and st.budget == budget
                         and st.metric == metric)
                jitter = np.linspace(-width * 0.3, width * 0.3, len(values))
                axis.scatter(x + jitter, values, s=9, color=color, alpha=0.25, linewidths=0)
                axis.errorbar(
                    x, s.mean, yerr=[[s.mean - s.ci_low], [s.ci_high - s.mean]],
                    fmt=marker, color=color, markersize=7, elinewidth=2, capsize=4,
                    markeredgecolor="white", markeredgewidth=1,
                    label=name if b_index == 0 else None, zorder=3,
                )
        axis.set_title(label, fontsize=11)
        axis.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        axis.set_xlabel("Simulations per run")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in list(axes.flat)[len(METRICS):]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False)
    fig.text(0.5, 0.01, f"Marker = mean, bar = {CONFIDENCE:.0%} bootstrap CI, faint dots = individual runs. "
             "No failure found counts as budget + 1 simulations to first failure.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_curves(paths: list[str | Path]) -> dict[int, dict[str, np.ndarray]]:
    """Return ``{budget: {method: array(seeds, budget)}}`` from coverage_curve.csv files."""
    curves: dict[int, dict[str, dict[int, list[tuple[int, float]]]]] = {}
    for path in paths:
        folder = Path(path) if Path(path).is_dir() else Path(path).parent
        csv_path = folder / "coverage_curve.csv"
        if not csv_path.is_file():
            continue
        per_run: dict[tuple[str, int], list[tuple[int, float]]] = {}
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                per_run.setdefault((row["method"], int(row["seed"])), []).append(
                    (int(row["evaluation"]), float(row["failure_coverage"])))
        for (method, seed), points in per_run.items():
            budget = max(e for e, _ in points)
            curves.setdefault(budget, {}).setdefault(method, {})[seed] = sorted(points)
    result: dict[int, dict[str, np.ndarray]] = {}
    for budget, methods in curves.items():
        result[budget] = {
            method: np.array([[v for _, v in seeds[s]] for s in sorted(seeds)])
            for method, seeds in methods.items()
        }
    return result


def plot_curves(curves: dict[int, dict[str, np.ndarray]], resamples: int, seed: int, path: Path) -> None:
    rng = np.random.default_rng(seed + 2)
    budgets = sorted(curves)
    fig, axes = plt.subplots(1, len(budgets), figsize=(6 * len(budgets), 4.3), squeeze=False)
    alpha = (1 - CONFIDENCE) / 2
    for axis, budget in zip(axes.flat, budgets):
        methods = method_order(curves[budget])
        for method in methods:
            name, color, _ = method_style(method, methods)
            values = curves[budget][method]
            x = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            low, high = _curve_ci(values, resamples, rng, alpha)
            axis.plot(x, mean, color=color, linewidth=2, label=name)
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            axis.annotate(name, (x[-1], mean[-1]), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=9, color="#333333")
        axis.set_title(f"{budget} simulations per run (n = {values.shape[0]} seeds)", fontsize=11)
        axis.set_xlabel("Simulations used")
        axis.set_ylabel("Distinct failure categories")
        axis.set_xlim(1, budget * 1.18)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper left")
    fig.text(0.5, 0.01, f"Line = mean over seeds, band = {CONFIDENCE:.0%} bootstrap CI of the mean.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _curve_ci(values: np.ndarray, resamples: int, rng, alpha: float, chunk: int = 1000):
    n = values.shape[0]
    means = []
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        weights = np.zeros((size, n))
        draws = rng.integers(0, n, size=(size, n))
        np.add.at(weights, (np.arange(size)[:, None], draws), 1)
        means.append(weights @ values / n)
    stacked = np.vstack(means)
    return np.quantile(stacked, alpha, axis=0), np.quantile(stacked, 1 - alpha, axis=0)


# --------------------------------------------------------------------------- entry point


def run_analysis(
    inputs: list[str | Path],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    reference: str = DEFAULT_REFERENCE,
    resamples: int = 10_000,
    seed: int = 0,
    overwrite: bool = False,
) -> tuple[list[SummaryStat], list[PairedTest]]:
    """Run the full analysis and write every output file. Returns the statistics."""
    if resamples < 100:
        raise ValueError("use at least 100 bootstrap resamples")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"{output} is not empty; choose a new folder or pass overwrite")
    runs = load_runs(inputs)
    stats = summarize(runs, resamples, seed)
    tests = paired_tests(runs, reference, resamples, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_tables(stats, tests, output)
    plot_metrics(runs, stats, output / "metrics_ci.png")
    curves = load_curves(inputs)
    if curves:
        plot_curves(curves, resamples, seed, output / "coverage_curve_ci.png")
    (output / "analysis_settings.json").write_text(json.dumps({
        "inputs": [str(p) for p in inputs],
        "reference": reference,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "confidence": CONFIDENCE,
        "ci_method": "percentile bootstrap of the mean",
        "test": "two-sided Wilcoxon signed-rank, paired by seed, zero differences dropped",
        "multiple_comparisons": "Holm over all paired tests in paired_tests.csv",
        "censoring": "first_failure_evaluation = budget + 1 when no failure was found",
        "versions": {name: version(name) for name in ("numpy", "matplotlib")},
    }, indent=1) + "\n")
    return stats, tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path,
                        help="result folders (or summary.csv files), e.g. one per budget")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="method compared against all others (default: diversity_aware)")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap random seed")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow writing into a non-empty output folder")
    args = parser.parse_args()
    stats, tests = run_analysis(args.inputs, args.output_dir, args.reference,
                                args.resamples, args.seed, args.overwrite)
    print(f"wrote {len(stats)} summary rows and {len(tests)} paired tests to {args.output_dir}")


if __name__ == "__main__":
    main()
