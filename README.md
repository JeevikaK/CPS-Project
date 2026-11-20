# CPS Project: Diversity-Aware AEB Falsification

This repository contains a dependency-light prototype for testing an autonomous emergency braking (AEB) system. It compares random testing, robustness-guided search, and diversity-aware falsification using a failure archive.

## Run

From the repository root:

```bash
python3 -m aeb_falsification --scenario aeb_falsification/example_scenario.json
python3 -m unittest discover -s tests -v
python3 -m aeb_falsification.diversity_demo
python3 -m aeb_falsification.experiment_runner --budget 100 --seeds 10
```

The current implementation uses a deterministic one-dimensional simulator. CARLA integration will be added through the same simulation/result interface.
