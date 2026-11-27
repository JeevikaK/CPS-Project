# Preliminary comparison after the foundation fixes

> Historical: the diversity-aware search was later replaced. See `../archive_guided/`.

100 simulations per method × seeds 0–9. **Not final:** settings are not frozen and
these seeds were used during development. Final runs need fresh seeds.

| Method | Failure categories (mean ± SD) | Mean worst impact (km/h) | Mean collisions per 100 |
|---|---:|---:|---:|
| Random | 14.4 ± 1.4 | 37.6 | 22.4 |
| CMA-ES | 9.7 ± 2.6 | 44.5 | 77.5 |
| Diversity-aware | 13.9 ± 2.1 | 40.8 | 31.6 |

The diversity-aware method does not exceed random coverage. It still samples mostly
at random and is not CMA-ES with archive guidance. These are descriptive results,
not significance claims, and are not comparable with earlier result folders: the
controller, TTC limit, scenario space and archive categories all changed.

- `failures.json`: worst scenario of each category in each run, with its outcome
- `trajectories/`: trajectory of the most severe failure of each type
- `coverage_curve.*`: coverage against simulations used
- `settings.json`: settings, written by the runner

Check the saved failures with:

```bash
python -m aeb_falsification.replay experiment_results/foundation_fixes/failures.json
```
