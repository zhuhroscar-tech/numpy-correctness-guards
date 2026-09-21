"""CLI tests for numpy-correctness-guards."""
from __future__ import annotations

import json

import pytest

from numpy_correctness_guards.cli import main


def test_list_json_includes_migrated_guard(capsys):
    rc = main(["list", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert "choice-shuffle" in payload
    assert "einsum-newdtype" in payload


def test_choice_shuffle_detect_json_exits_0_or_1_and_has_required_fields(capsys):
    rc = main([
        "run", "choice-shuffle", "detect", "--json",
        "--population-size", "300", "--sample-size", "50",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "numpy_version" in payload
    assert "affected" in payload
    assert "detail" in payload
    assert rc in (0, 1)
    assert rc == (1 if payload["affected"] else 0)


def test_choice_shuffle_verify_json_reports_passed_field(capsys):
    rc = main([
        "run", "choice-shuffle", "verify", "--json",
        "--population-size", "50", "--sample-size", "10",
        "--trials", "50", "--tolerance", "0.15",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["guard"] == "choice-shuffle"
    assert "passed" in payload
    assert "max_abs_freq_diff" in payload
    assert rc == (0 if payload["passed"] else 1)


def test_einsum_newdtype_detect_without_quaddtype_reports_clean_error(capsys, monkeypatch):
    """Missing optional probe dependency should be a structured error, not a false clean bill."""
    import numpy_correctness_guards.cli as cli_mod

    def _raise_import_error():
        raise ImportError("no module named numpy_quaddtype (simulated)")

    monkeypatch.setattr(cli_mod, "detect_einsum_newstyle_dtype_bug", _raise_import_error)
    rc = main(["run", "einsum-newdtype", "detect", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "error" in payload
    assert rc == 2


def test_einsum_newdtype_detect_text_mode_prints_status_headline(capsys, monkeypatch):
    import numpy_correctness_guards.cli as cli_mod

    def _raise_import_error():
        raise ImportError("no module named numpy_quaddtype (simulated)")

    monkeypatch.setattr(cli_mod, "detect_einsum_newstyle_dtype_bug", _raise_import_error)
    rc = main(["run", "einsum-newdtype", "detect", "--no-color"])
    out = capsys.readouterr().out
    assert "cannot probe: numpy_quaddtype not installed" in out
    assert "numpy_quaddtype is required" in out
    assert rc == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "numpy-correctness-guards" in out
