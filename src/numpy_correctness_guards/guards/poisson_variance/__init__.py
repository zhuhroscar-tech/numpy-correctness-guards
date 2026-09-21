"""Poisson large-lambda variance guard."""
from __future__ import annotations

from .core import (
    PoissonVarianceDiagnosis,
    diagnose,
    safe_poisson,
    stable_log_pmf,
)

__all__ = [
    "PoissonVarianceDiagnosis",
    "diagnose",
    "safe_poisson",
    "stable_log_pmf",
]
