"""Failure archive used by the diversity-aware falsifier."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Scenario, SimulationResult


@dataclass(frozen=True)
class FailureSignature:
    failure_type: str
    severity_bin: int
    friction_bin: str = "unknown"
    sensor_bin: str = "unknown"


@dataclass(frozen=True)
class ArchiveEntry:
    signature: FailureSignature
    result: SimulationResult
    scenario: Scenario | None = None


def make_failure_signature(
    result: SimulationResult,
    scenario: Scenario | None = None,
) -> FailureSignature:
    """Map a result and its operating conditions to an archive cell."""
    failure_type = result.failure_type or "safe"
    if failure_type == "collision":
        # Impact speed in 5 m/s steps; ego speed never exceeds 20 m/s.
        severity_bin = min(3, int(result.impact_speed // 5.0))
    elif failure_type == "unsafe_separation":
        # Shortfall below the 2 m minimum in 0.5 m steps.
        severity_bin = min(3, int(-result.metrics.separation_robustness // 0.5))
    elif failure_type == "unsafe_ttc":
        # Shortfall in 0.1 s steps; larger shortfalls also break separation.
        severity_bin = min(2, int(-result.metrics.ttc_robustness // 0.1))
    else:
        severity_bin = 0
    if scenario is None:
        friction_bin = "unknown"
        sensor_bin = "unknown"
    else:
        friction_bin = "low_friction" if scenario.road_friction < 0.6 else "normal_friction"
        sensor_bin = (
            "degraded_sensor"
            if scenario.sensor_delay >= 0.25 or scenario.sensor_noise_std >= 0.5
            else "clean_sensor"
        )
    return FailureSignature(failure_type, severity_bin, friction_bin, sensor_bin)


class FailureArchive:
    """Keep one representative result for each discovered failure cell."""

    def __init__(self) -> None:
        self._entries: dict[FailureSignature, ArchiveEntry] = {}

    def record(
        self,
        result: SimulationResult,
        scenario: Scenario | None = None,
    ) -> tuple[FailureSignature, bool]:
        """Store a failure; safe outcomes are never stored or novel."""
        signature = make_failure_signature(result, scenario)
        if result.failure_type is None:
            return signature, False
        is_new = signature not in self._entries
        current = self._entries.get(signature)
        if current is None or result.search_score < current.result.search_score:
            self._entries[signature] = ArchiveEntry(signature, result, scenario)
        return signature, is_new

    @property
    def coverage(self) -> int:
        return len(self._entries)

    def contains(self, signature: FailureSignature) -> bool:
        return signature in self._entries

    @property
    def signatures(self) -> tuple[FailureSignature, ...]:
        return tuple(
            sorted(
                self._entries,
                key=lambda item: (
                    item.failure_type,
                    item.severity_bin,
                    item.friction_bin,
                    item.sensor_bin,
                ),
            )
        )

    def entries(self) -> tuple[ArchiveEntry, ...]:
        return tuple(self._entries.values())
