# Archive ablation (100 simulations per arm, 30 seeds: 6000–6029)

## Noise seed: `per_evaluation`

| Arm | Failure categories (mean ± SD) | Worst impact (km/h) | Collisions / run | Restarts | Archive-seeded | Failures that survive reseeding |
|---|---:|---:|---:|---:|---:|---:|
| Random (reference) | 14.0 ± 2.0 | 37.7 | 21.3 | 0.0 | 0.0 | 94% |
| Archive mutation search (reference) | 15.9 ± 2.8 | 44.2 | 49.3 | 0.0 | 0.0 | 95% |
| CMA-ES, no stall restarts | 10.8 ± 2.7 | 44.1 | 75.8 | 0.0 | 0.0 | 97% |
| CMA-ES (no archive) | 10.8 ± 2.8 | 44.0 | 73.5 | 0.2 | 0.0 | 96% |
| + archive novelty fitness | 13.3 ± 3.4 | 42.9 | 57.6 | 0.7 | 0.0 | 95% |
| + archive restarts | 10.8 ± 2.8 | 44.0 | 74.4 | 0.2 | 0.1 | 97% |
| + both (archive-guided CMA-ES) | 13.1 ± 4.4 | 42.1 | 61.8 | 0.7 | 0.5 | 96% |

Paired difference against plain CMA-ES (95% bootstrap CI; Holm-corrected p):

| Arm | Δ categories | Δ worst impact (km/h) |
|---|---|---|
| Random (reference) | +3.2 [+2.0, +4.3], p=0.0003 | -6.3 [-7.5, -5.2], p=0.0003 |
| Archive mutation search (reference) | +5.1 [+4.0, +6.2], p=0.0003 | +0.1 [-0.8, +1.1], p=1 |
| CMA-ES, no stall restarts | -0.0 [-0.4, +0.3], p=1 | +0.1 [+0.0, +0.2], p=1 |
| + archive novelty fitness | +2.5 [+1.1, +3.9], p=0.0084 | -1.2 [-2.4, -0.1], p=0.23 |
| + archive restarts | +0.0 [-0.2, +0.3], p=1 | -0.0 [-0.1, +0.0], p=1 |
| + both (archive-guided CMA-ES) | +2.3 [+0.7, +3.9], p=0.032 | -1.9 [-3.7, -0.4], p=0.11 |

## Noise seed: `fixed`

| Arm | Failure categories (mean ± SD) | Worst impact (km/h) | Collisions / run | Restarts | Archive-seeded | Failures that survive reseeding |
|---|---:|---:|---:|---:|---:|---:|
| Random (reference) | 14.2 ± 2.1 | 37.7 | 20.8 | 0.0 | 0.0 | 95% |
| Archive mutation search (reference) | 16.4 ± 2.2 | 44.0 | 48.5 | 0.0 | 0.0 | 96% |
| CMA-ES, no stall restarts | 11.0 ± 3.2 | 43.7 | 74.1 | 0.0 | 0.0 | 96% |
| CMA-ES (no archive) | 11.4 ± 3.1 | 43.7 | 71.9 | 0.2 | 0.0 | 96% |
| + archive novelty fitness | 13.6 ± 2.9 | 42.4 | 57.5 | 0.7 | 0.0 | 95% |
| + archive restarts | 11.4 ± 2.9 | 43.7 | 72.6 | 0.2 | 0.2 | 96% |
| + both (archive-guided CMA-ES) | 13.1 ± 2.9 | 42.5 | 61.6 | 0.7 | 0.5 | 96% |

Paired difference against plain CMA-ES (95% bootstrap CI; Holm-corrected p):

| Arm | Δ categories | Δ worst impact (km/h) |
|---|---|---|
| Random (reference) | +2.8 [+1.4, +4.2], p=0.0025 | -6.0 [-7.2, -4.9], p=0.0003 |
| Archive mutation search (reference) | +5.0 [+3.8, +6.2], p=0.0003 | +0.3 [-0.6, +1.5], p=1 |
| CMA-ES, no stall restarts | -0.5 [-1.0, -0.1], p=0.12 | +0.0 [+0.0, +0.0], p=1 |
| + archive novelty fitness | +2.2 [+1.0, +3.3], p=0.004 | -1.3 [-2.6, +0.1], p=0.39 |
| + archive restarts | -0.0 [-0.5, +0.4], p=1 | +0.0 [+0.0, +0.0], p=1 |
| + both (archive-guided CMA-ES) | +1.7 [+0.5, +2.8], p=0.026 | -1.2 [-2.5, +0.2], p=0.42 |

Categories are failure-only archive cells (outcome × severity bin × friction × sensor region). Impact is ego speed at contact; 0 when a run has no collision. Reseeding re-simulates each archived worst failure with 5 fresh noise seeds.
