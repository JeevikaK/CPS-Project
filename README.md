# CPS Project: Diversity-Aware AEB Falsification

This repository contains a dependency-light prototype for testing an autonomous emergency braking (AEB) system. It compares random testing, robustness-guided search, and diversity-aware falsification using a failure archive.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python3 -m aeb_falsification --scenario aeb_falsification/example_scenario.json
python3 -m unittest discover -s tests -v
python3 -m aeb_falsification.diversity_demo
python3 -m aeb_falsification.experiment_runner --budget 100 --seeds 10
python3 -m aeb_falsification.analysis experiment_results/archive_guided --output-dir experiment_results/latest/analysis
```

The current scope uses a reproducible one-dimensional simulator; CARLA is not required. See `aeb_falsification/README.md` for model assumptions, safety definitions, and validation.

The latest results are in `experiment_results/archive_guided/` (preliminary). Runs go to
`experiment_results/latest/` (git-ignored) unless `--output-dir` is given. Saved failures can be
checked with `python3 -m aeb_falsification.replay <folder>/failures.json`.

## Statistics and confidence intervals

`aeb_falsification.analysis` turns one or more result folders (for example one per
budget) into report-ready statistics and plots:

```bash
python3 -m aeb_falsification.analysis experiment_results/archive_guided \
    --output-dir experiment_results/latest/analysis
```

It writes `summary_stats.csv` (mean, SD, median, 95% percentile-bootstrap CI per
budget/method/metric), `paired_tests.csv` (diversity-aware minus each other method,
paired by seed: mean difference with bootstrap CI, two-sided Wilcoxon signed-rank,
Holm-adjusted p-values over all tests, rank-biserial effect size), `results_table.md`,
`metrics_ci.png`, `coverage_curve_ci.png` and `analysis_settings.json`. Runs with no
failure count as budget + 1 simulations to first failure. Use `--reference` to change
the comparison method. Existing non-empty output folders are not overwritten unless
`--overwrite` is given.

The older top-level experiment results were rerun after the environment corrections with 100 simulations per method and seeds 0–9. Previous results are preserved in `experiment_results/before_environment_fixes/`.


## CMA-ES baseline

`robustness_guided_search` now uses pycma 4.6.0 with normalized [0, 1]
coordinates, built-in bounded sampling, and initial step size 0.2. It evaluates
exactly the requested simulation budget, including a final partial population.
Complete populations update the mean, covariance, and step size; if the optimizer
stops early, it restarts within the remaining budget. Seeds use a local NumPy
generator, including seed 0, without changing global random state.

The objective is unchanged: minimize aggregate safety robustness for non-collisions
and `-10 - impact_speed` for collisions. This is a severity-aware objective, not
formal robustness alone. All existing methods use that score. The diversity
method mutates archived failures; it is not CMA-ES with archive guidance, so
comparisons do not isolate archive effects.

The saved top-level experiment results use the previous local-search baseline;
they do not measure this CMA-ES implementation. Write new runs to a separate
folder, for example:

```bash
MPLBACKEND=Agg python -m aeb_falsification.experiment_runner --budget 100 --seeds 10 --output-dir experiment_results/my_run
```

Implementation follows the [pycma ask/tell interface](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html).


The 100-budget, 10-seed comparison from before the foundation fixes is in `experiment_results/cma_es/`; it used the old controller, TTC limit, scenario space and archive categories. Coverage excludes safe outcomes, impact speed is in km/h, and plot dots show individual runs. `settings.json` documents settings and metric definitions. Earlier top-level results remain historical.
