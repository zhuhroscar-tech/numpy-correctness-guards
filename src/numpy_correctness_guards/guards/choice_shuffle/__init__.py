"""Guard for numpy Generator.choice shuffle handling."""

from .core import (
    BugDetectionResult,
    detect_shuffle_ignored_bug,
    independent_reference_weighted_sample_without_replacement,
    safe_weighted_choice,
)

__all__ = [
    "BugDetectionResult",
    "detect_shuffle_ignored_bug",
    "independent_reference_weighted_sample_without_replacement",
    "safe_weighted_choice",
]
