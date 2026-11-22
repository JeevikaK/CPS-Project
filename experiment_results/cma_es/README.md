# CMA-ES comparison

100 simulations per method × 10 seeds × 3 methods = 3,000 simulations.

| Method | Mean failure categories | Mean worst impact (km/h) | Mean collisions per 100 |
|---|---:|---:|---:|
| Random | 24.1 | 59.14 | 41.4 |
| CMA-ES | 13.8 | 65.49 | 87.4 |
| Diversity-aware | 22.9 | 62.22 | 51.9 |

Bars and numbers show averages; dots show individual runs. Safe outcomes are excluded from coverage. Impact speed is ego speed at contact.

Random has the highest mean coverage; CMA-ES has the highest mean worst impact speed. These are descriptive findings, not significance claims. Diversity-aware search still uses its original local/global proposals and internal novelty logic, so differences do not isolate archive effects. Categories represent outcomes and operating conditions, not proven causes.

Earlier top-level results remain historical. See settings.json for settings and metric definitions.
