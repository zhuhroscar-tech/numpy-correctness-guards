"""Tests for the timedelta64 floor-division guard."""
from __future__ import annotations

import numpy as np

from numpy_correctness_guards.guards.timedelta64_floordiv import (
    BugDetectionResult,
    detect_negative_floordiv_truncation_bug,
    safe_timedelta64_floordiv,
    verify_workaround_against_oracle,
)


def test_detect_reports_internally_consistent_live_result():
    result = detect_negative_floordiv_truncation_bug()
    assert isinstance(result, BugDetectionResult)
    assert result.numpy_version == np.__version__
    assert result.total_checked == 7 * 81 * 7
    assert result.mismatches > 0 if result.affected else result.mismatches == 0
    assert result.detail


def test_detect_isolates_positive_dividends_as_unaffected():
    positive_mismatches = 0
    for value in range(1, 41):
        for divisor in (2, 3, 5, 7):
            got = np.timedelta64(value, "us") // divisor
            want = np.timedelta64(value // divisor, "us")
            if got != want:
                positive_mismatches += 1
    assert positive_mismatches == 0


def test_control_timedelta64_div_timedelta64_path_is_unaffected():
    delta = np.timedelta64(-7, "us")
    divisor_td = np.timedelta64(2, "us")
    assert delta // divisor_td == -4


def test_detect_reports_affected_with_forced_buggy_behavior(monkeypatch):
    import math
    import numpy_correctness_guards.guards.timedelta64_floordiv.core as core_module

    def _buggy_floordiv(td, divisor):
        raw = int(td.astype(np.int64))
        unit = np.datetime_data(td.dtype)[0]
        return np.timedelta64(math.trunc(raw / divisor), unit)

    monkeypatch.setattr(core_module, "_numpy_floordiv", _buggy_floordiv)
    result = detect_negative_floordiv_truncation_bug(
        units=("us",), divisors=(2, 3), values=(-7, -5, -1, 1, 7)
    )
    assert result.affected is True
    assert result.mismatches > 0
    assert "CONFIRMED" in result.detail


def test_detect_reports_not_affected_when_behavior_is_fixed(monkeypatch):
    import numpy_correctness_guards.guards.timedelta64_floordiv.core as core_module

    def _fixed_floordiv(td, divisor):
        raw = int(td.astype(np.int64))
        unit = np.datetime_data(td.dtype)[0]
        return np.timedelta64(raw // divisor, unit)

    monkeypatch.setattr(core_module, "_numpy_floordiv", _fixed_floordiv)
    result = detect_negative_floordiv_truncation_bug(
        units=("us",), divisors=(2, 3, -2), values=(-7, -5, -1, 1, 7)
    )
    assert result.affected is False
    assert result.mismatches == 0
    assert "Not reproduced" in result.detail


def test_safe_floordiv_matches_python_floor_oracle_when_forced():
    for unit in ("ns", "us", "ms", "s"):
        for value in range(-30, 31):
            for divisor in (2, 3, 5, 7, -2, -3, -5, -7):
                delta = np.timedelta64(value, unit)
                got = safe_timedelta64_floordiv(delta, divisor, force_workaround=True)
                want = np.timedelta64(value // divisor, unit)
                assert got == want, f"{unit} {value} // {divisor}: got {got}, want {want}"


def test_safe_floordiv_passes_through_when_workaround_disabled():
    delta = np.timedelta64(-7, "us")
    assert safe_timedelta64_floordiv(delta, 2, force_workaround=False) == delta // 2


def test_safe_floordiv_handles_nat_passthrough():
    nat = np.timedelta64("NaT", "us")
    assert np.isnat(safe_timedelta64_floordiv(nat, 2, force_workaround=True))
    assert np.isnat(safe_timedelta64_floordiv(nat, 2, force_workaround=False))
    assert np.isnat(safe_timedelta64_floordiv(nat, 2))


def test_safe_floordiv_auto_detects_and_applies_workaround_by_default():
    delta = np.timedelta64(-7, "us")
    assert safe_timedelta64_floordiv(delta, 2) == np.timedelta64(-7 // 2, "us")


def test_verify_workaround_against_oracle_passes():
    payload = verify_workaround_against_oracle()
    assert payload["passed"] is True
    assert payload["failures"] == 0
    assert payload["checked"] == 7 * 81 * 7
