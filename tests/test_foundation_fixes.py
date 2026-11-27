"""Regression checks for controller latching, avoidable scenarios, archive, and replay."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aeb_falsification import Scenario, SimulationConfig, simulate
from aeb_falsification.archive import FailureArchive, make_failure_signature
from aeb_falsification.controller import controller_step
from aeb_falsification.experiment_runner import run_experiment
from aeb_falsification.models import ControllerConfig, ControllerMode
from aeb_falsification.replay import replay, write_trajectory
from aeb_falsification.search.common import random_vector, vector_to_scenario
from aeb_falsification.simulator import minimum_avoidable_distance


def sample(count):
    rng = np.random.default_rng(42)
    return [vector_to_scenario(random_vector(rng), i) for i in range(count)]


class ControllerTests(unittest.TestCase):
    def test_safe_on_dry_road_with_ideal_sensors(self):
        for scenario in sample(200):
            ideal = dataclasses.replace(scenario, road_friction=1., sensor_delay=0., sensor_noise_std=0.)
            # Friction changed, so keep the gap avoidable.
            gap = minimum_avoidable_distance(ideal.ego_speed, ideal.pedestrian_speed, 1.)
            self.assertGreaterEqual(ideal.pedestrian_distance, gap)
            self.assertIsNone(simulate(ideal).failure_type, ideal)

    def test_safe_at_closest_avoidable_gap(self):
        for speed in np.linspace(8, 20, 13):
            for walking in (0., 1., 2.):
                gap = minimum_avoidable_distance(speed, walking, 1.)
                result = simulate(Scenario(speed, max(8., gap + 1e-6), walking, 1., 0., 0., 0))
                self.assertIsNone(result.failure_type, (speed, walking))

    def test_does_not_creep_into_stationary_pedestrian(self):
        # Previously crept to 1.96 m under perfect conditions.
        result = simulate(Scenario(12., 40., 0., 1., 0., 0., 0), SimulationConfig(horizon=60))
        self.assertIsNone(result.failure_type)
        self.assertEqual(result.trace.ego_speed[-1], 0.)

    def test_latch_and_release(self):
        config, emergency = ControllerConfig(), ControllerMode.EMERGENCY_BRAKING
        self.assertEqual(controller_step(50, 5, 0, config, emergency)[1], emergency)
        self.assertEqual(controller_step(5, 0, 0, config, emergency)[1], ControllerMode.WARNING)
        self.assertEqual(controller_step(50, 0, 0, config, emergency)[1], ControllerMode.CRUISING)

    def test_follows_again_after_pedestrian_moves_on(self):
        result = simulate(Scenario(12., 40., 1.5, 1., 0., 0., 0), SimulationConfig(horizon=40))
        self.assertIsNone(result.failure_type)
        self.assertGreater(result.trace.ego_position[-1], 60.)


class AvoidableScenarioTests(unittest.TestCase):
    def test_ideal_full_braking_is_always_safe(self):
        def full_brake(distance, speed, walking, config, previous):
            return config.emergency_braking * config.friction_factor, ControllerMode.EMERGENCY_BRAKING
        with patch("aeb_falsification.simulator.controller_step", full_brake):
            for scenario in sample(300):
                self.assertIsNone(simulate(scenario).failure_type, scenario)

    def test_bound_is_tight(self):
        def full_brake(distance, speed, walking, config, previous):
            return config.emergency_braking * config.friction_factor, ControllerMode.EMERGENCY_BRAKING
        with patch("aeb_falsification.simulator.controller_step", full_brake):
            for friction in (.35, .6, 1.):
                gap = minimum_avoidable_distance(20., 0., friction)
                self.assertIsNotNone(simulate(Scenario(20., gap - .2, 0., friction, 0., 0., 0)).failure_type)

    def test_failures_remain_to_be_found(self):
        rate = np.mean([simulate(s).failure_type is not None for s in sample(300)])
        self.assertGreater(rate, .1)
        self.assertLess(rate, .7)


class ArchiveTests(unittest.TestCase):
    def test_safe_outcome_is_not_stored_or_novel(self):
        scenario = Scenario(8, 60, 2, 1, 0, 0, 1)
        archive = FailureArchive()
        self.assertFalse(archive.record(simulate(scenario), scenario)[1])
        self.assertEqual(archive.coverage, 0)

    def test_entry_keeps_scenario(self):
        scenario = Scenario(20, 8, 0, .35, .5, 0, 1)
        archive = FailureArchive()
        archive.record(simulate(scenario), scenario)
        self.assertEqual(archive.entries()[0].scenario, scenario)

    def test_severity_bins_use_the_violated_quantity(self):
        result = simulate(Scenario(20, 8, 0, .35, .5, 0, 1))
        for kind, metric, values in (
            ("unsafe_separation", "separation_robustness", {-.1: 0, -.6: 1, -1.2: 2, -1.9: 3}),
            ("unsafe_ttc", "ttc_robustness", {-.05: 0, -.15: 1, -.25: 2}),
        ):
            for margin, expected in values.items():
                result.failure_type = kind
                result.metrics = dataclasses.replace(result.metrics, **{metric: margin})
                self.assertEqual(make_failure_signature(result).severity_bin, expected)


class ReplayTests(unittest.TestCase):
    def test_saved_failures_replay_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_experiment(budget=15, seeds=(0, 1), output_dir=directory)
            for name in ("settings.json", "coverage_curve.csv", "coverage_curve.png"):
                self.assertTrue(Path(directory, name).exists())
            data = json.loads(Path(directory, "failures.json").read_text())
            self.assertTrue(data["failures"])
            config = SimulationConfig(**data["simulation"])
            for saved in data["failures"]:
                self.assertTrue(replay(saved, config)[1])
            result, _ = replay(saved, config)
            path = Path(directory, "trace.csv")
            write_trajectory(result, path)
            self.assertEqual(len(path.read_text().splitlines()), len(result.trace.time) + 1)
            saved["outcome"]["min_distance"] += 1.
            self.assertFalse(replay(saved, config)[1])


if __name__ == "__main__":
    unittest.main()
