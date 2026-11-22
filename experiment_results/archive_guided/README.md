# Preliminary comparison with archive-guided diversity search

100 simulations per method × seeds 0–9. **Not final:** settings are not frozen and
these seeds were used during development.

| Method | Failure categories (mean ± SD) | Mean worst impact (km/h) | Mean collisions per 100 |
|---|---:|---:|---:|
| Random | 14.4 ± 1.4 | 37.6 | 22.4 |
| CMA-ES | 9.7 ± 2.6 | 44.5 | 77.5 |
| Diversity-aware | 16.6 ± 2.5 | 44.5 | 45.7 |

Diversity-aware search now mutates archived failures; about 29% of its samples here
are uniform random, against 88% for the version in `../foundation_fixes/`. Random and
CMA-ES are unchanged. It is not CMA-ES with archive guidance, so a same-optimizer
ablation is still needed.

Settings were chosen on seeds 3000–3029 and checked on seeds 4000–4029 (30 runs).
Mean failure categories (mean worst impact in km/h):

| Budget | Random | CMA-ES | Diversity-aware | Diversity vs random coverage, paired Wilcoxon |
|---|---:|---:|---:|---:|
| 100 | 13.9 (37.5) | 10.8 (44.9) | 16.9 (43.8) | p = 3.5e-4 |
| 300 | 22.5 (40.4) | 11.4 (47.2) | 26.5 (45.5) | p = 2.0e-5 |
| 1000 | 29.4 (42.8) | 11.4 (47.6) | 33.2 (46.5) | p = 1.6e-6 |

P-values are not corrected for multiple comparisons. CMA-ES still finds slightly more
severe collisions. Seeds 0–9, 1000–1029, 3000–3029 and 4000–4029 are used; final
evaluation should use others, such as 2000–2029.
