import unittest

from aeb_falsification import Scenario, simulate
from aeb_falsification.archive import FailureArchive, make_failure_signature
from aeb_falsification.search import diversity_aware_search


def collision_scenario(speed: float, seed: int) -> Scenario:
    return Scenario(
        ego_speed=speed,
        pedestrian_distance=8.0,
        pedestrian_speed=0.0,
        road_friction=0.35,
        sensor_delay=0.5,
        sensor_noise_std=0.0,
        seed=seed,
    )


class ArchiveAndDiversityTests(unittest.TestCase):
    def test_archive_distinguishes_collision_speed_bins(self):
        slow = simulate(collision_scenario(12.0, 1))
        fast = simulate(collision_scenario(20.0, 1))
        self.assertNotEqual(make_failure_signature(slow), make_failure_signature(fast))

    def test_archive_includes_operating_region(self):
        scenario = collision_scenario(20.0, 1)
        result = simulate(scenario)
        signature = make_failure_signature(result, scenario)
        self.assertEqual(signature.friction_bin, "low_friction")
        self.assertEqual(signature.sensor_bin, "degraded_sensor")

    def test_archive_keeps_new_cells(self):
        archive = FailureArchive()
        result = simulate(collision_scenario(20.0, 1))
        scenario = collision_scenario(20.0, 1)
        _, first_new = archive.record(result, scenario)
        _, second_new = archive.record(result, scenario)
        self.assertTrue(first_new)
        self.assertFalse(second_new)
        self.assertEqual(archive.coverage, 1)

    def test_diversity_search_reports_coverage(self):
        result = diversity_aware_search(budget=20, seed=3)
        self.assertEqual(result.evaluations, 20)
        self.assertGreaterEqual(result.archive_coverage, 1)
        self.assertEqual(result.method, "diversity_aware")


if __name__ == "__main__":
    unittest.main()
