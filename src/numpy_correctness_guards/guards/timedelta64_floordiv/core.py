"""Core logic for the timedelta64 floor-division guard.

Guards numpy/numpy#32522: ``numpy.timedelta64(v, unit) // int`` can
truncate toward zero for negative values instead of flooring toward negative
infinity. The detector compares NumPy's live result against an independent
Python-integer floor-division oracle, and the workaround re-wraps the raw
integer quotient in the original timedelta64 unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_TEST_UNITS = ("ns", "us", "ms", "s", "m", "h", "D")
_TEST_DIVISORS = (2, 3, 5, 7, -2, -3, -5)
_TEST_VALUES = tuple(range(-40, 41))


@dataclass
class BugDetectionResult:
    """Result of probing the installed NumPy for this bug."""

    numpy_version: str
    affected: bool
    detail: str
    mismatches: int
    total_checked: int


def _reference_floordiv_timedelta64(value: int, unit: str, divisor: int) -> "np.timedelta64":
    """Independent oracle using Python integer floor-division semantics."""
    return np.timedelta64(value // divisor, unit)


def _numpy_floordiv(td: "np.timedelta64", divisor: int) -> "np.timedelta64":
    """The NumPy operation under test, kept as a monkeypatchable seam."""
    return td // divisor


def detect_negative_floordiv_truncation_bug(
    *,
    units=_TEST_UNITS,
    divisors=_TEST_DIVISORS,
    values=_TEST_VALUES,
) -> BugDetectionResult:
    """Probe installed NumPy for ``timedelta64 // int`` truncation.

    The check is version-independent and sweeps units, divisors, and values so
    partial upstream fixes are still visible.
    """
    total = 0
    mismatches = 0
    first_mismatch = None

    for unit in units:
        for value in values:
            for divisor in divisors:
                total += 1
                td = np.timedelta64(value, unit)
                got = _numpy_floordiv(td, divisor)
                want = _reference_floordiv_timedelta64(value, unit, divisor)
                if got != want:
                    mismatches += 1
                    if first_mismatch is None:
                        first_mismatch = (unit, value, divisor, str(got), str(want))

    if mismatches == 0:
        return BugDetectionResult(
            numpy_version=np.__version__,
            affected=False,
            detail=(
                f"Not reproduced: all {total} (unit, value, divisor) combinations "
                "matched Python's floor-division oracle exactly. "
                "numpy/numpy#32522 appears fixed here; "
                "safe_timedelta64_floordiv() will pass calls straight through."
            ),
            mismatches=0,
            total_checked=total,
        )

    assert first_mismatch is not None
    unit, value, divisor, got, want = first_mismatch
    return BugDetectionResult(
        numpy_version=np.__version__,
        affected=True,
        detail=(
            f"CONFIRMED: {mismatches}/{total} combinations diverge from "
            f"Python floor-division semantics. First divergence: "
            f"np.timedelta64({value}, {unit!r}) // {divisor} = {got}, "
            f"expected {want} (numpy/numpy#32522)."
        ),
        mismatches=mismatches,
        total_checked=total,
    )


def safe_timedelta64_floordiv(
    delta: "np.timedelta64",
    divisor: int,
    *,
    force_workaround: Optional[bool] = None,
) -> "np.timedelta64":
    """Floor-divide ``delta`` by an integer with correct negative semantics.

    ``force_workaround`` controls behavior:
    - ``True`` always applies the Python-integer oracle path.
    - ``False`` delegates directly to NumPy.
    - ``None`` probes the installed NumPy for this call's unit/divisor and
      applies the workaround only when needed.
    """
    if np.isnat(delta):
        return delta

    unit = np.datetime_data(delta.dtype)[0]

    if force_workaround is None:
        probe = detect_negative_floordiv_truncation_bug(
            units=(unit,),
            divisors=(divisor if divisor != 0 else 1,),
            values=(-7, -1, 1, 7),
        )
        needs_workaround = probe.affected
    else:
        needs_workaround = force_workaround

    if not needs_workaround:
        return _numpy_floordiv(delta, divisor)

    raw = int(delta.astype(np.int64))
    return np.timedelta64(raw // divisor, unit)


def verify_workaround_against_oracle() -> dict:
    """Verify the workaround across the same sweep used by the CLI."""
    checked = 0
    failures = []
    for unit in _TEST_UNITS:
        for value in _TEST_VALUES:
            for divisor in _TEST_DIVISORS:
                checked += 1
                delta = np.timedelta64(value, unit)
                got = safe_timedelta64_floordiv(delta, divisor, force_workaround=True)
                want = np.timedelta64(value // divisor, unit)
                if got != want:
                    failures.append((unit, value, divisor, str(got), str(want)))

    return {
        "checked": checked,
        "failures": len(failures),
        "passed": len(failures) == 0,
        "sample_failures": failures[:5],
    }
