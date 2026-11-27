# Archive ablation (300 simulations per arm, 30 seeds: 6000–6029)

## Noise seed: `per_evaluation`

| Arm | Failure categories (mean ± SD) | Worst impact (km/h) | Collisions / run | Restarts | Archive-seeded | Failures that survive reseeding |
|---|---:|---:|---:|---:|---:|---:|
| Random (reference) | 21.8 ± 1.9 | 40.8 | 67.1 | 0.0 | 0.0 | 95% |
| Archive mutation search (reference) | 26.1 ± 2.5 | 45.5 | 140.7 | 0.0 | 0.0 | 96% |
| CMA-ES, no stall restarts | 11.4 ± 3.4 | 46.9 | 274.3 | 0.0 | 0.0 | 97% |
| CMA-ES (no archive) | 17.4 ± 3.7 | 46.4 | 232.1 | 1.4 | 0.0 | 96% |
| + archive novelty fitness | 20.6 ± 3.6 | 45.8 | 210.4 | 2.0 | 0.0 | 96% |
| + archive restarts | 15.1 ± 3.2 | 46.3 | 253.1 | 1.3 | 0.9 | 96% |
| + both (archive-guided CMA-ES) | 19.8 ± 4.2 | 45.4 | 220.7 | 2.0 | 1.3 | 96% |

Paired difference against plain CMA-ES (95% bootstrap CI; Holm-corrected p):

| Arm | Δ categories | Δ worst impact (km/h) |
|---|---|---|
| Random (reference) | +4.5 [+3.1, +5.9], p=0.0003 | -5.5 [-6.3, -4.8], p=0.0003 |
| Archive mutation search (reference) | +8.7 [+7.3, +10.1], p=0.0003 | -0.9 [-1.4, -0.4], p=0.0062 |
| CMA-ES, no stall restarts | -5.9 [-7.2, -4.8], p=0.0003 | +0.5 [+0.3, +0.7], p=0.0003 |
| + archive novelty fitness | +3.2 [+1.3, +5.1], p=0.0083 | -0.6 [-1.1, -0.1], p=0.07 |
| + archive restarts | -2.3 [-3.5, -1.0], p=0.0063 | -0.1 [-0.5, +0.3], p=0.72 |
| + both (archive-guided CMA-ES) | +2.4 [+0.2, +4.5], p=0.042 | -1.0 [-1.6, -0.4], p=0.0088 |

Categories are failure-only archive cells (outcome × severity bin × friction × sensor region). Impact is ego speed at contact; 0 when a run has no collision. Reseeding re-simulates each archived worst failure with 5 fresh noise seeds.
