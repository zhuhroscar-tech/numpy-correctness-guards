"""SeedSequence.spawn thread-safety guard."""
from __future__ import annotations

from .core import (
    BugDetectionResult,
    GuardedSeedSequence,
    detect_spawn_race,
    verify_guard_eliminates_race,
)

__all__ = [
    "BugDetectionResult",
    "GuardedSeedSequence",
    "detect_spawn_race",
    "verify_guard_eliminates_race",
]
