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

This writes to `experiment_results/latest/` unless `--output-dir` is given:
`summary.csv`/`.png`, `coverage_curve.csv`/`.png`, `settings.json`, and `failures.json`.

Every saved failure can be re-simulated and checked, and trajectories exported:

```bash
python3 -m aeb_falsification.replay experiment_results/latest/failures.json --trajectories traj
```


## Validated environment assumptions

This is a one-dimensional point-object model: the pedestrian moves ahead of the
vehicle in the same direction at constant speed. It does not model road crossing,
vehicle geometry, perception, or real-world vehicle certification. All distances
are metres, speeds m/s, accelerations m/s², and times seconds.

The controller samples observations and holds acceleration between updates.
Negative acceleration in every controller mode is multiplied by
`min(road_friction, 1)`. Friction is a simplified braking multiplier rather than a
validated tyre model. Motion uses constant-acceleration integration, with speed
clamped at zero and no extra travel after stopping. Braking is latched: it holds
until the vehicle is no longer closing and the gap is at least `resume_distance`
(default 10 m). Without this, the vehicle crept into the 2 m limit even with
perfect sensors on a dry road.

Search covers only avoidable scenarios. `minimum_avoidable_distance` gives the
smallest initial gap at which ideal full braking from t=0 meets every requirement;
`vector_to_scenario` rescales the distance coordinate to start there (distance upper
bound 80 m). A violation is therefore never forced by the initial conditions.

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
- TTC safety: always `TTC >= minimum_ttc` (default 1.0 s, below the controller's
  1.5 s emergency trigger so correct braking is not a violation). TTC is infinite when
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
The suite contains 42 tests, including analytic stopping distance, analytic contact
speed at three time steps, a known safe encounter, adverse friction and delay,
TTC-only violations, threshold equality, and zero-noise comparisons at 0.05 and
0.025 s. These checks validate the intended simplified implementation, not a
real vehicle model.

The archive stores failures only. Severity bins are 5 m/s of impact speed (4 bins),
0.5 m of separation shortfall (4), and 0.1 s of TTC shortfall (3), each split by
friction and sensor region. In 20,000 random scenarios 34 of the 44 cells occurred;
about 35% of scenarios failed, and none failed on a dry road with ideal sensors in
the tests.

Results in `experiment_results/` other than `foundation_fixes/` predate the
controller, TTC, scenario-space and archive changes and are kept as history; their
counts are not comparable. Archive guidance of CMA-ES remains pending work, and one
noise seed is still reused for a whole run. See the root README for CMA-ES details.
