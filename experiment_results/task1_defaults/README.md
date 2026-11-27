# Three-method comparison with the Task 1 defaults

100 simulations × seeds 0–9, rerun after adding CMA-ES stall restarts and per-simulation
noise seeds. This is the same comparison as `../archive_guided/`, with nothing else changed.

| Method | Categories (mean ± SD) | Worst impact (km/h) | Collisions / 100 |
|---|---:|---:|---:|
| Random | 14.4 ± 2.4 (was 14.4 ± 1.4) | 37.4 (37.6) | 21.6 (22.4) |
| CMA-ES | 10.0 ± 2.7 (was 9.7 ± 2.6) | 43.3 (44.5) | 75.8 (77.5) |
| Diversity-aware (mutation) | 16.4 ± 2.4 (was 16.6 ± 2.5) | 43.7 (44.5) | 45.6 (45.7) |

All changes are well within one SD. All 408 saved failures replay exactly
(`python -m aeb_falsification.replay experiment_results/task1_defaults/failures.json`).
Preliminary, like `../archive_guided/`: seeds 0–9 were used during development.
