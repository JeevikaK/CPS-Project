"""Deterministic one-dimensional closed-loop AEB simulator."""

from __future__ import annotations

from math import ceil, inf
from dataclasses import asdict

import numpy as np

from .controller import compute_ttc, controller_step
from .models import (
    ControllerConfig,
    ControllerMode,
    Scenario,
    SimulationConfig,
    SimulationResult,
    Trace,
)
from .monitor import classify_failure, safety_metrics, violated_properties


def _friction_factor(road_friction: float) -> float:
    return min(road_friction, 1.0)


def minimum_avoidable_distance(
    ego_speed: float,
    pedestrian_speed: float,
    road_friction: float,
    config: SimulationConfig | None = None,
) -> float:
    """Smallest initial gap at which ideal full braking from t=0 is fully safe.

    Ideal means perfect sensing and no reaction time. Below this gap no
    controller can satisfy the separation and TTC requirements, so a violation
    there says nothing about the controller under test.
    """
    config = config or SimulationConfig()
    closing = ego_speed - pedestrian_speed
    if closing <= 0.0:
        return config.minimum_separation
    deceleration = -config.emergency_braking * _friction_factor(road_friction)
    # With final gap g, TTC while braking is g/c + c/(2a); bound it for c <= closing.
    worst = min(closing, deceleration * config.minimum_ttc)
    ttc_gap = config.minimum_ttc * worst - worst ** 2 / (2.0 * deceleration)
    return closing ** 2 / (2.0 * deceleration) + max(config.minimum_separation, ttc_gap)


def simulate(
    scenario: Scenario,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run one deterministic closed-loop simulation."""
    config = config or SimulationConfig()
    numeric_values = [value for value in asdict(config).values()] + [
        value for key, value in asdict(scenario).items() if key != "seed"
    ]
    if not all(np.isfinite(value) for value in numeric_values):
        raise ValueError("scenario and configuration values must be finite")
    if config.distance_scale <= 0 or config.ttc_scale <= 0:
        raise ValueError("robustness reference scales must be positive")
    if not (config.emergency_braking <= config.normal_braking < 0):
        raise ValueError("emergency braking must be at least as strong as normal braking")
    if not (0 <= config.minimum_ttc <= config.emergency_ttc <= config.warning_ttc):
        raise ValueError("invalid TTC thresholds")
    if not (0 <= config.collision_distance < config.minimum_separation):
        raise ValueError("invalid distance thresholds")
    if config.maximum_acceleration < 0 or config.target_speed < 0:
        raise ValueError("target speed and maximum acceleration must be non-negative")
    if config.dt <= 0.0 or config.horizon <= 0.0:
        raise ValueError("dt and horizon must be positive")
    if scenario.pedestrian_distance <= config.collision_distance:
        raise ValueError("pedestrian_distance must be positive")
    if scenario.ego_speed < 0.0 or scenario.pedestrian_speed < 0.0:
        raise ValueError("initial speeds must be non-negative")
    if scenario.road_friction <= 0.0:
        raise ValueError("road_friction must be positive")
    if not 0 <= scenario.sensor_delay <= config.maximum_sensor_delay or scenario.sensor_noise_std < 0.0:
        raise ValueError("sensor delay and noise must be non-negative")

    steps = int(ceil(config.horizon / config.dt))
    time = np.minimum(np.arange(steps + 1, dtype=float) * config.dt, config.horizon)
    ego_position = np.zeros(steps + 1, dtype=float)
    ego_speed = np.zeros(steps + 1, dtype=float)
    pedestrian_position = np.zeros(steps + 1, dtype=float)
    measured_distance = np.zeros(steps + 1, dtype=float)
    true_distance = np.zeros(steps + 1, dtype=float)
    ttc = np.full(steps + 1, inf, dtype=float)
    brake_command = np.zeros(steps + 1, dtype=float)
    controller_mode = np.empty(steps + 1, dtype=object)

    ego_speed[0] = scenario.ego_speed
    pedestrian_position[0] = scenario.pedestrian_distance
    rng = np.random.default_rng(scenario.seed)
    true_history: list[float] = []
    friction = _friction_factor(scenario.road_friction)
    collision_detected = False
    impact_speed = 0.0
    mode = ControllerMode.CRUISING
    controller_config = ControllerConfig(
        target_speed=config.target_speed,
        warning_ttc=config.warning_ttc,
        emergency_ttc=config.emergency_ttc,
        minimum_separation=config.minimum_separation,
        resume_distance=config.resume_distance,
        maximum_acceleration=config.maximum_acceleration,
        normal_braking=config.normal_braking,
        emergency_braking=config.emergency_braking,
        friction_factor=friction,
    )

    for i in range(steps + 1):
        true_distance[i] = pedestrian_position[i] - ego_position[i]
        true_history.append(float(true_distance[i]))
        delayed_distance = float(np.interp(max(0.0, time[i] - scenario.sensor_delay), time[:i + 1], true_history))
        noise = rng.normal(0.0, scenario.sensor_noise_std)
        measured_distance[i] = max(0.0, delayed_distance + noise)
        ttc[i] = compute_ttc(
            true_distance[i], ego_speed[i], scenario.pedestrian_speed
        )

        command, mode = controller_step(
            measured_distance[i],
            ego_speed[i],
            scenario.pedestrian_speed,
            controller_config,
            mode,
        )
        brake_command[i] = command
        controller_mode[i] = mode.value

        if i == steps:
            break

        duration = time[i + 1] - time[i]
        # Hold acceleration over the sample, integrating only until speed is zero.
        moving_duration = min(duration, -ego_speed[i] / command) if command < 0 else duration

        def gap_at(elapsed):
            moving = min(elapsed, moving_duration)
            travel = ego_speed[i] * moving + 0.5 * command * moving ** 2
            return true_distance[i] + scenario.pedestrian_speed * elapsed - travel

        # Check closest approach as well as the endpoint: a moving pedestrian
        # can pull away again after the ego vehicle has slowed within this step.
        candidates = [0.0, duration, moving_duration]
        if command != 0:
            turning = (scenario.pedestrian_speed - ego_speed[i]) / command
            if 0 < turning < moving_duration:
                candidates.append(turning)
        closest_time = min(candidates, key=gap_at)
        if gap_at(closest_time) <= config.collision_distance:
            low, high = 0.0, closest_time
            for _ in range(60):
                middle = (low + high) / 2
                if gap_at(middle) > config.collision_distance:
                    low = middle
                else:
                    high = middle
            contact_time = high
            impact_speed = max(0.0, float(ego_speed[i] + command * min(contact_time, moving_duration)))
            collision_detected = True
            contact_position = pedestrian_position[i] + scenario.pedestrian_speed * contact_time
            # Terminal contact is padded to retain the existing fixed-size interface.
            pedestrian_position[i + 1:] = contact_position
            ego_position[i + 1:] = contact_position - config.collision_distance
            ego_speed[i + 1:] = 0.0
            true_distance[i + 1:] = config.collision_distance
            measured_distance[i + 1:] = config.collision_distance
            ttc[i + 1:] = 0.0
            brake_command[i + 1:] = 0.0
            controller_mode[i + 1:] = "terminal_collision"
            break
        ego_speed[i + 1] = max(0.0, ego_speed[i] + command * duration)
        ego_position[i + 1] = ego_position[i] + ego_speed[i] * moving_duration + 0.5 * command * moving_duration ** 2
        pedestrian_position[i + 1] = pedestrian_position[i] + scenario.pedestrian_speed * duration

    collision = collision_detected or bool(np.any(true_distance <= config.collision_distance))
    trace = Trace(
        time=time,
        ego_position=ego_position,
        ego_speed=ego_speed,
        pedestrian_position=pedestrian_position,
        measured_distance=measured_distance,
        true_distance=true_distance,
        ttc=ttc,
        brake_command=brake_command,
        controller_mode=controller_mode,
        collision=collision,
        impact_speed=impact_speed,
    )
    metrics = safety_metrics(trace, config)
    failure_type, severity = classify_failure(
        trace,
        metrics,
        config,
        road_friction=scenario.road_friction,
        sensor_delay=scenario.sensor_delay,
        sensor_noise_std=scenario.sensor_noise_std,
    )
    finite_ttc = ttc[np.isfinite(ttc)]
    min_ttc = float(np.min(finite_ttc)) if finite_ttc.size else inf
    return SimulationResult(
        trace=trace,
        min_distance=float(np.min(true_distance)),
        min_ttc=min_ttc,
        collision=collision,
        failure_type=failure_type,
        severity=severity,
        metrics=metrics,
        impact_speed=impact_speed,
        operating_conditions=tuple(label for label, present in (
            ("low_friction", scenario.road_friction < 0.6),
            ("sensor_delay", scenario.sensor_delay > 0),
            ("sensor_noise", scenario.sensor_noise_std > 0),
        ) if present),
        violated_properties=violated_properties(trace, metrics),
    )
