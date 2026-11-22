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
    previous_mode: ControllerMode = ControllerMode.CRUISING,
) -> tuple[float, ControllerMode]:
    """Return an acceleration command and the current hybrid controller mode.

    Braking is latched: once engaged it holds until the vehicle is no longer
    closing and the gap has reopened to ``resume_distance``. Emergency braking
    never relaxes to normal braking while still closing.
    """
    distance = max(0.0, float(measured_distance))
    speed = max(0.0, float(ego_speed))
    ttc = compute_ttc(distance, speed, pedestrian_speed)

    closing = speed > pedestrian_speed
    emergency_latched = previous_mode == ControllerMode.EMERGENCY_BRAKING and closing
    braking_latched = previous_mode != ControllerMode.CRUISING and (
        closing or distance < config.resume_distance
    )

    if distance <= config.minimum_separation or ttc <= config.emergency_ttc or emergency_latched:
        return config.emergency_braking * config.friction_factor, ControllerMode.EMERGENCY_BRAKING

    if ttc <= config.warning_ttc or braking_latched:
        return config.normal_braking * config.friction_factor, ControllerMode.WARNING

    speed_error = config.target_speed - speed
    acceleration = min(config.maximum_acceleration, max(config.normal_braking, 0.8 * speed_error))
    return acceleration * config.friction_factor if acceleration < 0 else acceleration, ControllerMode.CRUISING
