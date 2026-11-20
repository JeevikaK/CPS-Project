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


def make_failure_signature(
    result: SimulationResult,
    scenario: Scenario | None = None,
) -> FailureSignature:
    """Map a result and its operating conditions to an archive cell."""
    failure_type = result.failure_type or "safe"
    if failure_type == "collision":
        # Collision cells distinguish low-, medium-, and high-speed impact.
        severity_bin = min(4, int(result.impact_speed // 5.0))
    elif failure_type == "safe":
        severity_bin = 0
    else:
        # Non-collision violations are binned by separation shortfall.
        severity_bin = min(4, int(max(0.0, result.severity) // 0.5))
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
        signature = make_failure_signature(result, scenario)
        is_new = signature not in self._entries
        current = self._entries.get(signature)
        if current is None or result.search_score < current.result.search_score:
            self._entries[signature] = ArchiveEntry(signature, result)
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
