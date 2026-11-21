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

`result.trace` contains the complete trajectory. `result.metrics` contains quantitative robustness values. `result.failure_type` is `None`, `collision`, `unsafe_separation`, or `unsafe_ttc`, in that priority order. `result.violated_properties` reports every violated property. `result.operating_conditions` separately records low friction, sensor delay, and noise; these tags do not establish causation.

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


## Validated environment assumptions

This is a one-dimensional point-object model: the pedestrian moves ahead of the
vehicle in the same direction at constant speed. It does not model road crossing,
vehicle geometry, perception, or real-world vehicle certification. All distances
are metres, speeds m/s, accelerations m/s², and times seconds.

The controller samples observations and holds acceleration between updates.
Negative acceleration in every controller mode is multiplied by
`min(road_friction, 1)`. Friction is a simplified braking multiplier rather than a
validated tyre model. Motion uses constant-acceleration integration, with speed
clamped at zero and no extra travel after stopping. The controller may resume
acceleration if its conditions permit; braking is not latched.

Delayed distance is interpolated from the true history; before simulation start,
the initial distance is held constant. Noise is independent Gaussian distance
noise per sample, reproducible for a fixed seed and time step. Ego and pedestrian
speeds are assumed known perfectly. Default allowed delay is 0–0.5 s. Changing the
time step changes the noise realization, so convergence checks use zero noise.
The simulation ends exactly at the configured horizon.

Contact is detected within each held-acceleration interval, including a possible
closest approach before its endpoint. Impact speed is the ego speed at first
contact (not relative impact speed). After contact, terminal state padding retains
the fixed-length trace interface; padded speed and commands are zero and mode is
`terminal_collision`. These samples are not post-crash dynamics.

## Safety properties

Over the simulated horizon, the sampled monitors evaluate:

- Collision avoidance: always `true_distance > collision_distance` (strict).
  Contact is a violation even though its distance robustness is exactly zero.
- Minimum separation: always `true_distance >= minimum_separation` (default 2 m).
- TTC safety: always `TTC >= emergency_ttc` (default 1.5 s). TTC is infinite when
  the vehicle is not closing. This is a conservative safety requirement, separate
  from the controller's emergency trigger; braking activation does not excuse it.

Individual robustness values are the minimum signed margin in metres or seconds.
The aggregate takes the minimum after division by explicit reference scales
(default 1 m and 1 s), making it dimensionless. Negative separation/TTC margins
violate their properties; equality satisfies them. Always use the collision flag
or `violated_properties` as well as aggregate robustness at strict boundaries.
Collision detection runs within steps; separation and TTC monitoring is sampled,
so these are numerical finite-trace checks, not continuous-time safety proofs.
Collision severity for search remains `-10 - impact_speed`, separate from formal
robustness. The legacy `severity` value combines separation shortfall and weighted
TTC shortfall and must not be interpreted as physical impact severity.

## Validation

Run `MPLBACKEND=Agg python3 -m unittest discover -s tests -v`.
The suite contains 22 tests, including analytic stopping distance, analytic contact
speed at three time steps, a known safe encounter, adverse friction and delay,
TTC-only violations, threshold equality, and zero-noise comparisons at 0.05 and
0.025 s. These checks validate the intended simplified implementation, not a
real vehicle model.

The files `experiment_results/summary.csv` and `summary.png` were regenerated after
these corrections with budget 100 and seeds 0–9. Original outputs are preserved in
`experiment_results/before_environment_fixes/`. Coverage still includes safe cells;
the robustness baseline is still local mutation search, not CMA-ES. Failure labels
also changed, so old and new coverage counts are not directly comparable. Search
algorithms, archive definitions, and CMA-ES implementation remain pending work.
