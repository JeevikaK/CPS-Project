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
python3 -m aeb_falsification.ablation   # archive ablation (same CMA-ES with/without archive)
```

The current scope uses a reproducible one-dimensional simulator; CARLA is not required. See `aeb_falsification/README.md` for model assumptions, safety definitions, and validation.

The latest results are in `experiment_results/archive_guided/` (preliminary). Runs go to
`experiment_results/latest/` (git-ignored) unless `--output-dir` is given. Saved failures can be
checked with `python3 -m aeb_falsification.replay <folder>/failures.json`.

The older top-level experiment results were rerun after the environment corrections with 100 simulations per method and seeds 0–9. Previous results are preserved in `experiment_results/before_environment_fixes/`.


## CMA-ES baseline

`robustness_guided_search` now uses pycma 4.6.0 with normalized [0, 1]
coordinates, built-in bounded sampling, and initial step size 0.2. It evaluates
exactly the requested simulation budget, including a final partial population.
Complete populations update the mean, covariance, and step size; if the optimizer
stops early, it restarts within the remaining budget. Seeds use a local NumPy
generator, including seed 0, without changing global random state.

It also restarts when stuck: when the generation-median fitness has not improved
(relative tolerance 1e-3) for 4 complete generations, or when all step sizes are
below 1e-3. `search/cma_es.py` holds this optimizer; `robustness_guided_search` is
its no-archive configuration.

The objective is unchanged: minimize aggregate safety robustness for non-collisions
and `-10 - impact_speed` for collisions. This is a severity-aware objective, not
formal robustness alone. All existing methods use that score. The diversity
method (`diversity_aware_search`) mutates archived failures; it is not CMA-ES, so the
three-way comparison alone does not isolate archive effects. The archive ablation does.

The saved top-level experiment results use the previous local-search baseline;
they do not measure this CMA-ES implementation. Write new runs to a separate
folder, for example:

```bash
MPLBACKEND=Agg python -m aeb_falsification.experiment_runner --budget 100 --seeds 10 --output-dir experiment_results/my_run
```

Implementation follows the [pycma ask/tell interface](https://cma-es.github.io/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html).


The 100-budget, 10-seed comparison from before the foundation fixes is in `experiment_results/cma_es/`; it used the old controller, TTC limit, scenario space and archive categories. Coverage excludes safe outcomes, impact speed is in km/h, and plot dots show individual runs. `settings.json` documents settings and metric definitions. Earlier top-level results remain historical.


## Archive-guided CMA-ES and ablation

`archive_guided_cma_search` is the same CMA-ES with two additions that read the
failure archive: a novelty bonus of 30 for reaching a new failure category (so it
outranks every known category), and restarts centred on the worst failure of the
least-visited category (70% of restarts; the rest are uniform). Switching both off
gives exactly `robustness_guided_search`.

```bash
python -m aeb_falsification.ablation                                   # 30 held-out seeds, budget 100
python -m aeb_falsification.ablation --noise per_evaluation fixed     --output-dir experiment_results/ablation                           # also the noise-seed study
```

It runs CMA-ES without stall restarts, the baseline, each archive component alone,
and both together, with random and archive-mutation search as references. It writes
`summary.csv`, `comparison.csv` (paired differences against the baseline, bootstrap
CIs, Holm-corrected permutation p-values), `ablation.png`, `results.md`, and
`settings.json`. Settings were tuned on seeds 5000–5029; the default evaluation
seeds are 6000–6029, and seeds 2000–2029 remain unused. Results are in
`experiment_results/ablation/` (budget 100) and `experiment_results/ablation_300/`.
The method write-up is `report/method.md`.

**Noise seed.** Every simulation now gets its own reproducible noise seed, derived
from the run seed and evaluation index (`noise="per_evaluation"`, the default for all
methods). Previously one seed was reused for a whole run, so every simulation saw the
same noise sequence. Pass `noise="fixed"` for the old behaviour. Saved scenarios carry
their seed, so replay is unaffected. `experiment_results/task1_defaults/` is the
three-method comparison rerun with these defaults.
