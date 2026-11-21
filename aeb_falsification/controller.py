"""Simple hybrid AEB controller used by the falsification experiments."""

from __future__ import annotations

from math import inf, isfinite

from .models import ControllerConfig, ControllerMode


def compute_ttc(distance: float, ego_speed: float, pedestrian_speed: float) -> float:
    """Return time-to-collision for a one-dimensional closing encounter."""
    closing_speed = ego_speed - pedestrian_speed
    if not isfinite(distance) or not isfinite(closing_speed) or closing_speed <= 0.0:
        return inf
    return distance / closing_speed


def controller_step(
    measured_distance: float,
    ego_speed: float,
    pedestrian_speed: float,
    config: ControllerConfig,
) -> tuple[float, ControllerMode]:
    """Return an acceleration command and the current hybrid controller mode."""
    distance = max(0.0, float(measured_distance))
    speed = max(0.0, float(ego_speed))
    ttc = compute_ttc(distance, speed, pedestrian_speed)

    if distance <= config.minimum_separation or ttc <= config.emergency_ttc:
        return config.emergency_braking * config.friction_factor, ControllerMode.EMERGENCY_BRAKING

    if ttc <= config.warning_ttc:
        return config.normal_braking * config.friction_factor, ControllerMode.WARNING

    speed_error = config.target_speed - speed
    acceleration = min(config.maximum_acceleration, max(config.normal_braking, 0.8 * speed_error))
    return acceleration * config.friction_factor if acceleration < 0 else acceleration, ControllerMode.CRUISING
