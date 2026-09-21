"""timedelta64 floor-division guard."""
from __future__ import annotations

from .core import (
    BugDetectionResult,
    detect_negative_floordiv_truncation_bug,
    safe_timedelta64_floordiv,
    verify_workaround_against_oracle,
)

__all__ = [
    "BugDetectionResult",
    "detect_negative_floordiv_truncation_bug",
    "safe_timedelta64_floordiv",
    "verify_workaround_against_oracle",
]
