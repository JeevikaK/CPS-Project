"""Independent physical checks and safety-boundary regression tests."""
import unittest
import numpy as np
from aeb_falsification import Scenario, SimulationConfig, simulate
from aeb_falsification.controller import controller_step
from aeb_falsification.models import ControllerConfig
from aeb_falsification.monitor import safety_metrics, classify_failure


class EnvironmentValidationTests(unittest.TestCase):
    def scenario(self, **changes):
        base = dict(ego_speed=10., pedestrian_distance=30., pedestrian_speed=0.,
                    road_friction=1., sensor_delay=0., sensor_noise_std=0., seed=1)
        return Scenario(**(base | changes))

    def test_friction_scales_all_braking_modes(self):
        for distance, speed in [(20, 10), (10, 10), (100, 20)]:
            dry, _ = controller_step(distance, speed, 0, ControllerConfig())
            wet, _ = controller_step(distance, speed, 0, ControllerConfig(friction_factor=.35))
            self.assertLess(dry, 0)
            self.assertAlmostEqual(wet, dry * .35)

    def test_constant_braking_stopping_distance(self):
        config = SimulationConfig(horizon=2, target_speed=0, minimum_separation=101, warning_ttc=100, emergency_ttc=100)
        result = simulate(self.scenario(pedestrian_distance=100), config)
        # v^2 / (2a) = 100 / 16 = 6.25 m; stopping time = 1.25 s.
        self.assertAlmostEqual(result.trace.ego_position[-1], 6.25, places=10)
        self.assertEqual(result.trace.ego_speed[-1], 0)

    def test_contact_speed_matches_analytic_solution(self):
        # v_contact^2 = v_initial^2 - 2*a*distance.
        scenario = self.scenario(pedestrian_distance=2)
        for dt in (.2, .05, .01):
            result = simulate(scenario, SimulationConfig(dt=dt))
            self.assertTrue(result.collision)
            self.assertAlmostEqual(result.impact_speed, np.sqrt(100 - 32), places=8)
            self.assertEqual(result.metrics.distance_robustness, 0)
            self.assertIn('collision_avoidance', result.violated_properties)

    def test_safe_stop_and_known_adverse_conditions(self):
        # Never resume, so only the first approach is compared.
        config = SimulationConfig(resume_distance=1000)
        safe = simulate(self.scenario(pedestrian_distance=40), config)
        slippery = simulate(self.scenario(pedestrian_distance=40, road_friction=.35), config)
        delayed = simulate(self.scenario(pedestrian_distance=40, sensor_delay=.5), config)
        known_safe = simulate(self.scenario(ego_speed=8, pedestrian_distance=60, pedestrian_speed=2))
        self.assertIsNone(known_safe.failure_type)
        self.assertFalse(known_safe.collision)
        self.assertLess(slippery.min_distance, safe.min_distance)
        self.assertLess(delayed.min_distance, safe.min_distance)
        self.assertIn('low_friction', slippery.operating_conditions)
        self.assertIn('sensor_delay', delayed.operating_conditions)

    def test_time_step_consistency(self):
        for scenario in [self.scenario(), self.scenario(road_friction=.35),
                         self.scenario(sensor_delay=.5), self.scenario(pedestrian_distance=2)]:
            coarse = simulate(scenario, SimulationConfig(dt=.05))
            fine = simulate(scenario, SimulationConfig(dt=.025))
            self.assertEqual(coarse.collision, fine.collision)
            self.assertLess(abs(coarse.min_distance - fine.min_distance), .6)
            self.assertLess(abs(coarse.impact_speed - fine.impact_speed), .6)

    def test_ttc_only_violation_and_threshold_equality(self):
        result = simulate(self.scenario())
        trace = result.trace
        trace.true_distance[:] = 3
        trace.ttc[:] = .9
        metrics = safety_metrics(trace, SimulationConfig())
        kind, _ = classify_failure(trace, metrics, SimulationConfig(), road_friction=.35,
                                  sensor_delay=.5, sensor_noise_std=1)
        self.assertEqual(kind, 'unsafe_ttc')
        trace.true_distance[:] = 2
        trace.ttc[:] = 1
        metrics = safety_metrics(trace, SimulationConfig())
        kind, _ = classify_failure(trace, metrics, SimulationConfig(), road_friction=1,
                                  sensor_delay=0, sensor_noise_std=0)
        self.assertIsNone(kind)

    def test_exact_horizon_and_invalid_inputs(self):
        result = simulate(self.scenario(), SimulationConfig(horizon=.12, dt=.05))
        self.assertEqual(result.trace.time[-1], .12)
        for scenario in [self.scenario(ego_speed=float('nan')), self.scenario(sensor_delay=.6)]:
            with self.assertRaises(ValueError):
                simulate(scenario)
