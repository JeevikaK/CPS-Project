import unittest

import numpy as np

from aeb_falsification import Scenario, SimulationConfig, simulate
from aeb_falsification.controller import compute_ttc, controller_step
from aeb_falsification.models import ControllerConfig, ControllerMode
from aeb_falsification.scenarios import sample_random_scenario


class AebSimulationTests(unittest.TestCase):
    def test_same_seed_is_deterministic(self):
        scenario = sample_random_scenario(11)
        first = simulate(scenario)
        second = simulate(scenario)
        np.testing.assert_array_equal(first.trace.true_distance, second.trace.true_distance)
        np.testing.assert_array_equal(first.trace.measured_distance, second.trace.measured_distance)
        self.assertEqual(first.failure_type, second.failure_type)

    def test_different_seeds_change_noise(self):
        base = sample_random_scenario(11)
        other = Scenario(**{**base.__dict__, "seed": 12})
        first = simulate(base)
        second = simulate(other)
        self.assertFalse(np.array_equal(first.trace.measured_distance, second.trace.measured_distance))

    def test_trace_has_expected_length_and_no_negative_speed(self):
        result = simulate(sample_random_scenario(2))
        expected = int(round(SimulationConfig().horizon / SimulationConfig().dt)) + 1
        self.assertEqual(len(result.trace.time), expected)
        self.assertGreaterEqual(float(np.min(result.trace.ego_speed)), 0.0)
        self.assertTrue(np.all(np.isfinite(result.trace.true_distance)))

    def test_controller_modes_progress_toward_emergency(self):
        config = ControllerConfig()
        modes = [
            controller_step(100.0, 15.0, 0.0, config)[1],
            controller_step(30.0, 15.0, 0.0, config)[1],
            controller_step(10.0, 15.0, 0.0, config)[1],
        ]
        self.assertEqual(modes, [ControllerMode.CRUISING, ControllerMode.WARNING, ControllerMode.EMERGENCY_BRAKING])

    def test_ttc_is_infinite_when_not_closing(self):
        self.assertTrue(np.isinf(compute_ttc(10.0, 1.0, 2.0)))

    def test_collision_case_is_classified(self):
        scenario = Scenario(ego_speed=20.0, pedestrian_distance=8.0, pedestrian_speed=0.0,
                            road_friction=0.35, sensor_delay=0.5, sensor_noise_std=0.0, seed=1)
        result = simulate(scenario)
        self.assertTrue(result.collision)
        self.assertEqual(result.failure_type, "collision")
        self.assertGreater(result.impact_speed, 0.0)
        self.assertLess(result.search_score, -10.0)

    def test_safe_case_has_no_collision(self):
        scenario = Scenario(ego_speed=8.0, pedestrian_distance=60.0, pedestrian_speed=2.0,
                            road_friction=1.0, sensor_delay=0.0, sensor_noise_std=0.0, seed=1)
        result = simulate(scenario)
        self.assertFalse(result.collision)
        self.assertIsNone(result.failure_type)
        self.assertEqual(result.impact_speed, 0.0)


if __name__ == "__main__":
    unittest.main()
