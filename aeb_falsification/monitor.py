"""Quantitative safety robustness and failure classification."""

from __future__ import annotations

from math import inf

import numpy as np

from .models import SafetyMetrics, SimulationConfig, Trace


def safety_metrics(trace: Trace, config: SimulationConfig) -> SafetyMetrics:
    """Compute min-based robustness values for the three safety requirements."""
    distance_robustness = float(np.min(trace.true_distance - config.collision_distance))
    separation_robustness = float(np.min(trace.true_distance - config.minimum_separation))
    finite_ttc = trace.ttc[np.isfinite(trace.ttc)]
    ttc_robustness = (
        float(np.min(finite_ttc - config.emergency_ttc))
        if finite_ttc.size
        else inf
    )
    overall = min(distance_robustness, separation_robustness, ttc_robustness)
    return SafetyMetrics(
        distance_robustness=distance_robustness,
        separation_robustness=separation_robustness,
        ttc_robustness=ttc_robustness,
        overall_robustness=float(overall),
    )


def classify_failure(
    trace: Trace,
    metrics: SafetyMetrics,
    config: SimulationConfig,
    *,
    road_friction: float,
    sensor_delay: float,
    sensor_noise_std: float,
) -> tuple[str | None, float]:
    """Classify a trace using the deterministic priority from the project plan."""
    min_distance = float(np.min(trace.true_distance))
    finite_ttc = trace.ttc[np.isfinite(trace.ttc)]
    min_ttc = float(np.min(finite_ttc)) if finite_ttc.size else inf
    severity = max(0.0, config.minimum_separation - min_distance)
    if np.isfinite(min_ttc):
        severity += max(0.0, config.emergency_ttc - min_ttc) * 0.1

    if trace.collision:
        return "collision", float(severity)

    emergency_indices = np.flatnonzero(
        trace.controller_mode == "emergency_braking"
    )
    if min_ttc <= config.emergency_ttc and emergency_indices.size == 0:
        return "late_braking", float(severity)

    if (
        metrics.separation_robustness < 0.0
        and road_friction < 0.99
    ):
        return "low_friction_failure", float(severity)

    if (
        metrics.separation_robustness < 0.0
        and (sensor_delay > 0.0 or sensor_noise_std > 0.0)
    ):
        return "sensor_delay_failure", float(severity)

    if metrics.separation_robustness < 0.0:
        return "near_miss", float(severity)

    return None, 0.0
