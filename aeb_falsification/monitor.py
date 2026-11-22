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
    overall = min(
        distance_robustness / config.distance_scale,
        separation_robustness / config.distance_scale,
        ttc_robustness / config.ttc_scale,
    )
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
    """Classify observed outcomes, without inferring causes from conditions."""
    min_distance = float(np.min(trace.true_distance))
    finite_ttc = trace.ttc[np.isfinite(trace.ttc)]
    min_ttc = float(np.min(finite_ttc)) if finite_ttc.size else inf
    severity = max(0.0, config.minimum_separation - min_distance)
    if np.isfinite(min_ttc):
        severity += max(0.0, config.emergency_ttc - min_ttc) * 0.1

    if trace.collision:
        return "collision", float(severity)

    if metrics.separation_robustness < 0.0:
        return "unsafe_separation", float(severity)
    if metrics.ttc_robustness < 0.0:
        return "unsafe_ttc", float(severity)
    return None, 0.0


def violated_properties(trace: Trace, metrics: SafetyMetrics) -> tuple[str, ...]:
    """Contact violates strict collision avoidance; other boundaries are safe."""
    return tuple(name for name, failed in (
        ("collision_avoidance", trace.collision or metrics.distance_robustness <= 0.0),
        ("minimum_separation", metrics.separation_robustness < 0.0),
        ("time_to_collision", metrics.ttc_robustness < 0.0),
    ) if failed)
