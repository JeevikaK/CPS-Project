# AEB Falsification Prototype

This package is the dependency-light closed-loop simulator for the CPS final project. It does not require CARLA.

From the workspace root, run:

```bash
python3 -m aeb_falsification --scenario aeb_falsification/example_scenario.json
```

The stable handoff interface is:

```python
from aeb_falsification import Scenario, simulate

result = simulate(Scenario(...))
```

`result.trace` contains the complete trajectory. `result.metrics` contains quantitative robustness values. `result.failure_type` is either `None` or a classified failure such as `collision`, `late_braking`, `low_friction_failure`, `sensor_delay_failure`, or `near_miss`.

For collisions, `result.impact_speed` preserves the speed at first contact and `result.search_score` provides a severity-aware objective. Formal robustness may still be `-2.0` at every collision because the separation specification reaches its boundary; search algorithms should use `search_score` when ranking scenarios. The archive also separates low/normal friction and clean/degraded sensor regions.

The first two search baselines use the same simulator interface and budget:

```bash
python3 -m aeb_falsification.search_demo
```

The diversity-aware search and archive can be run with:

```bash
python3 -m aeb_falsification.diversity_demo
```

For the actual equal-budget comparison across seeds:

```bash
python3 -m aeb_falsification.experiment_runner --budget 100 --seeds 10
```

This writes `experiment_results/summary.csv` and `experiment_results/summary.png`.
