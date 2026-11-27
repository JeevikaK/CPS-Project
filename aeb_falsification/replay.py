"""Re-simulate saved failures, check their outcomes, and export trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from math import isclose
from pathlib import Path

from .experiment_runner import outcome
from .models import Scenario, SimulationConfig, SimulationResult
from .simulator import simulate

TRACE_FIELDS = (
    "time", "ego_position", "ego_speed", "pedestrian_position", "true_distance",
    "measured_distance", "ttc", "brake_command", "controller_mode",
)


def replay(saved: dict, config: SimulationConfig) -> tuple[SimulationResult, bool]:
    """Return the replayed result and whether it matches the saved outcome."""
    result = simulate(Scenario(**saved["scenario"]), config)
    actual, expected = outcome(result), saved["outcome"]
    matches = actual.keys() == expected.keys() and all(
        isclose(value, expected[key], rel_tol=1e-6, abs_tol=1e-6)
        if isinstance(value, float) else value == expected[key]
        for key, value in actual.items()
    )
    return result, matches


def write_trajectory(result: SimulationResult, path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(TRACE_FIELDS)
        writer.writerows(zip(*(getattr(result.trace, name) for name in TRACE_FIELDS)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("failures", type=Path, help="failures.json from experiment_runner")
    parser.add_argument("--trajectories", type=Path, metavar="DIR",
                        help="write the most severe trajectory of each failure type here")
    args = parser.parse_args()
    data = json.loads(args.failures.read_text())
    config = SimulationConfig(**data["simulation"])

    worst: dict[str, SimulationResult] = {}
    mismatches = 0
    for index, saved in enumerate(data["failures"]):
        result, matches = replay(saved, config)
        if not matches:
            mismatches += 1
            print(f"mismatch at failure {index}: {saved['method']} seed {saved['seed']}")
        kind = result.failure_type
        if kind and (kind not in worst or result.search_score < worst[kind].search_score):
            worst[kind] = result
    print(f"replayed {len(data['failures'])} failures, {mismatches} mismatches")
    if args.trajectories:
        args.trajectories.mkdir(parents=True, exist_ok=True)
        for kind, result in worst.items():
            write_trajectory(result, args.trajectories / f"{kind}.csv")
    raise SystemExit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
