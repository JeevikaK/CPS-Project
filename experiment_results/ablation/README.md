# Archive ablation: same CMA-ES with and without the failure archive

30 held-out seeds (6000–6029), not used for tuning; settings were frozen on seeds
5000–5029. Budget 100 here (both noise-seed modes); budget 300 in `../ablation_300/`.
The tables are in `results.md` and the paired statistics in `comparison.csv`. Rerun with:

```bash
python -m aeb_falsification.ablation --noise per_evaluation fixed --output-dir experiment_results/ablation
python -m aeb_falsification.ablation --budget 300 --output-dir experiment_results/ablation_300
```

## Main numbers (per-simulation noise seeds)

Mean failure categories (mean worst impact, km/h), 30 runs per arm:

| Arm | Budget 100 | Budget 300 |
|---|---:|---:|
| Random (reference) | 14.0 (37.7) | 21.8 (40.8) |
| Archive mutation search (reference) | **15.9** (44.2) | **26.1** (45.5) |
| CMA-ES, no stall restarts | 10.8 (44.1) | 11.4 (**46.9**) |
| CMA-ES with stall restarts (baseline) | 10.8 (44.0) | 17.4 (46.4) |
| + archive novelty fitness | 13.3 (42.9) | 20.6 (45.8) |
| + archive-seeded restarts | 10.8 (44.0) | 15.1 (46.3) |
| + both (archive-guided CMA-ES) | 13.1 (42.1) | 19.8 (45.4) |

Paired against the baseline on the same seeds (95% bootstrap CI, Holm-corrected
permutation p):

| Effect on failure categories | Budget 100 | Budget 300 |
|---|---|---|
| Stall restarts (baseline vs no restarts) | +0.0 [−0.3, +0.4], p = 1 | **+5.9** [+4.8, +7.2], p < 0.001 |
| Novelty fitness | **+2.5** [+1.1, +3.9], p = 0.008 | **+3.2** [+1.3, +5.1], p = 0.008 |
| Archive-seeded restarts | +0.0 [−0.2, +0.3], p = 1 | **−2.3** [−3.5, −1.0], p = 0.006 |
| Both | **+2.3** [+0.7, +3.9], p = 0.03 | **+2.4** [+0.2, +4.5], p = 0.04 |

## What this shows

1. **The archive helps CMA-ES find more distinct failures**, and the novelty fitness
   is responsible for the gain. With the optimizer held fixed, novelty adds 2.5–3.2
   categories. It costs 1–2 km/h of worst-case impact speed, and runs contain fewer
   collisions.
2. **Archive-seeded restarts do not help.** They have no effect at budget 100, where
   few restarts happen (0.2 per run), and they reduce coverage at budget 300. Restarting
   on known failures pulls the search back into regions already covered. Novelty
   fitness with uniform restarts (20.6 at budget 300) does better than both
   components together (19.8). We kept the pre-registered configuration (both
   components) as `archive_guided_cma_search`. Switching to novelty only on the
   strength of these seeds would have to be confirmed on fresh seeds.
3. **Stall restarts matter once the budget allows them.** At budget 100 CMA-ES
   rarely stalls within about 11 generations. At budget 300, plain CMA-ES without
   restarts stays near 11 categories, while the restarted version reaches 17.4.
4. **Archive guidance does not make CMA-ES the best diversity method.**
   Archive-guided CMA-ES stays below uniform random search in coverage (13.1 vs 14.0
   at 100, 19.8 vs 21.8 at 300), and well below the archive mutation search. CMA-ES
   concentrates its samples by design. Its strength is severity: it finds the
   highest impact speeds in every configuration. For the research question, the
   archive mutation search remains the diversity-aware method to compare. Archive
   guidance is a measurable but smaller improvement on a severity optimizer.

## Noise-seed decision

All methods now use a fresh, reproducible noise seed for every simulation
(`per_evaluation`), derived from the run seed and evaluation index. Earlier, one seed
was reused for a whole run (`fixed`).

- **Results barely depend on the choice.** At budget 100 every arm changes by at most
  0.6 categories between modes, well inside the run-to-run SD of 2–4.
- **Failures do not depend on a particular noise draw.** Re-simulating each archived
  worst failure with 5 unrelated noise seeds reproduces a failure 94–97% of the time
  in both modes.
- **We chose per-simulation seeds** because they are the principled protocol. With a
  fixed seed every simulation in a run sees the same standard-normal sequence, so
  "30 seeds" sample only 30 noise realizations, and a search could tune failures to
  one of them. Each saved scenario stores its seed, so replay stays exact.
  `noise="fixed"` remains available.

## Caveats

These are 30 runs of a simplified one-dimensional model. Categories are outcome and
operating-condition cells, not proven causes. CMA-ES numbers are reproducible on one
platform but drift slightly across operating systems; see `report/method.md`.
