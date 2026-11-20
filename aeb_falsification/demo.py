"""Command-line demo for the first simulator milestone."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .scenarios import load_scenario, sample_random_scenario, save_scenario
from .simulator import simulate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", type=Path, default=Path("aeb_demo.png"))
    args = parser.parse_args()

    scenario = load_scenario(args.scenario) if args.scenario else sample_random_scenario(args.seed)
    if args.scenario is None:
        save_scenario(scenario, "example_scenario.json")
    result = simulate(scenario)

    print("Scenario:", scenario)
    print("collision:", result.collision)
    print("minimum distance:", f"{result.min_distance:.3f} m")
    print("minimum TTC:", f"{result.min_ttc:.3f} s")
    print("failure type:", result.failure_type)
    print("severity:", f"{result.severity:.3f}")
    print("overall robustness:", f"{result.metrics.overall_robustness:.3f}")

    trace = result.trace
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(trace.time, trace.true_distance, label="true distance")
    axes[0].plot(trace.time, trace.measured_distance, alpha=0.7, label="measured distance")
    axes[0].axhline(2.0, color="tab:red", linestyle="--", label="minimum separation")
    axes[0].set_ylabel("Distance (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(trace.time, trace.ego_speed, label="ego speed")
    axes[1].step(trace.time, trace.brake_command, where="post", label="control command")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed / acceleration")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print("plot:", args.plot)


if __name__ == "__main__":
    main()
