"""Deterministic one-dimensional closed-loop AEB simulator."""

from __future__ import annotations

from math import ceil, inf

import numpy as np

from .controller import compute_ttc, controller_step
from .models import (
    ControllerConfig,
    Scenario,
    SimulationConfig,
    SimulationResult,
    Trace,
)
from .monitor import classify_failure, safety_metrics


def _friction_factor(road_friction: float) -> float:
    return float(np.clip(road_friction / 1.0, 0.35, 1.0))


def simulate(
    scenario: Scenario,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run one deterministic closed-loop simulation."""
    config = config or SimulationConfig()
    if config.dt <= 0.0 or config.horizon <= 0.0:
        raise ValueError("dt and horizon must be positive")
    if scenario.pedestrian_distance <= 0.0:
        raise ValueError("pedestrian_distance must be positive")
    if scenario.ego_speed < 0.0 or scenario.pedestrian_speed < 0.0:
        raise ValueError("initial speeds must be non-negative")
    if scenario.road_friction <= 0.0:
        raise ValueError("road_friction must be positive")
    if scenario.sensor_delay < 0.0 or scenario.sensor_noise_std < 0.0:
        raise ValueError("sensor delay and noise must be non-negative")

    steps = int(round(config.horizon / config.dt))
    time = np.arange(steps + 1, dtype=float) * config.dt
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
    delay_steps = int(round(scenario.sensor_delay / config.dt))
    true_history: list[float] = []
    friction = _friction_factor(scenario.road_friction)
    collision_detected = False
    impact_speed = 0.0
    controller_config = ControllerConfig(
        target_speed=config.target_speed,
        warning_ttc=config.warning_ttc,
        emergency_ttc=config.emergency_ttc,
        minimum_separation=config.minimum_separation,
        maximum_acceleration=config.maximum_acceleration,
        normal_braking=config.normal_braking,
        emergency_braking=config.emergency_braking,
        friction_factor=friction,
    )

    for i in range(steps + 1):
        true_distance[i] = pedestrian_position[i] - ego_position[i]
        true_history.append(float(true_distance[i]))
        delayed_index = max(0, i - delay_steps)
        noise = rng.normal(0.0, scenario.sensor_noise_std)
        measured_distance[i] = max(0.0, true_history[delayed_index] + noise)
        ttc[i] = compute_ttc(
            true_distance[i], ego_speed[i], scenario.pedestrian_speed
        )

        command, mode = controller_step(
            measured_distance[i],
            ego_speed[i],
            scenario.pedestrian_speed,
            controller_config,
        )
        brake_command[i] = command
        controller_mode[i] = mode.value

        if i == steps:
            break

        ego_speed[i + 1] = max(0.0, ego_speed[i] + command * config.dt)
        ego_position[i + 1] = ego_position[i] + ego_speed[i] * config.dt
        pedestrian_position[i + 1] = (
            pedestrian_position[i] + scenario.pedestrian_speed * config.dt
        )

        # Stop the physical evolution at first contact. Keeping the remaining
        # trace samples at contact makes fixed-budget experiments easier to
        # compare without allowing the vehicles to pass through each other.
        if pedestrian_position[i + 1] - ego_position[i + 1] <= config.collision_distance:
            collision_detected = True
            impact_speed = float(ego_speed[i + 1])
            ego_position[i + 1] = pedestrian_position[i + 1] - config.collision_distance
            for j in range(i + 1, steps + 1):
                pedestrian_position[j] = pedestrian_position[i + 1]
                ego_position[j] = ego_position[i + 1]
                ego_speed[j] = 0.0
                true_distance[j] = config.collision_distance
                measured_distance[j] = config.collision_distance
                ttc[j] = 0.0
                brake_command[j] = config.emergency_braking * friction
                controller_mode[j] = "emergency_braking"
            break

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
    )
