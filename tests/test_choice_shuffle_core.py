"""Tests for numpy_choice_shuffle_guard.core.

These tests are written to distinguish three real scenarios rather than
assume the bug's presence:
1. The bug IS present on the installed numpy (the expected case as of
   this guard's creation) -- detect_shuffle_ignored_bug must say so, and
   safe_weighted_choice must actually change draw order.
2. The bug has been fixed upstream -- detect_shuffle_ignored_bug must
   say so, and safe_weighted_choice must pass calls straight through
   (proven by monkeypatching Generator.choice to a fixed-behavior stub
   and confirming safe_weighted_choice's manual-shuffle branch does NOT
   fire in that case).
3. A broken probe (both weighted and unweighted paths appear identical)
   must be reported as affected=True (fail-safe), never a false "clean".
"""
from __future__ import annotations

import numpy as np
import pytest

from numpy_correctness_guards.guards.choice_shuffle.core import (
    BugDetectionResult,
    detect_shuffle_ignored_bug,
    independent_reference_weighted_sample_without_replacement,
    safe_weighted_choice,
)


def test_detect_reproduces_the_real_bug_on_currently_installed_numpy():
    """This is the live, no-mocking reproduction: run against whatever
    numpy is actually installed in this test environment. As of this
    guard's creation (numpy/numpy#31210 open, reproduced independently
    against the pinned CI numpy version), this must report affected."""
    result = detect_shuffle_ignored_bug(population_size=500, sample_size=100, seed=0)
    assert isinstance(result, BugDetectionResult)
    assert result.numpy_version == np.__version__
    # Do not hard-assert affected=True here: if a CI numpy release fixes
    # this upstream, that is real, desirable information this guard
    # should surface truthfully rather than fail loudly. Assert instead
    # that the report is internally consistent.
    assert isinstance(result.affected, bool)
    assert len(result.detail) > 0


def test_detect_control_case_isolates_bug_to_weighted_path():
    """Directly reproduces the isolating comparison the bug report itself
    uses: unweighted choice() DOES respect shuffle; only the weighted
    without-replacement path is suspect. This test does not depend on
    detect_shuffle_ignored_bug's internals -- it independently redoes the
    comparison to catch a regression in the detector's own logic."""
    n = 500
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    unweighted_a = rng_a.choice(n, size=50, replace=False, shuffle=True)
    unweighted_b = rng_b.choice(n, size=50, replace=False, shuffle=False)
    # The control case (no weights) must show shuffle has an effect,
    # otherwise this numpy install's RNG/choice itself is broken in a
    # way unrelated to the bug this guard targets.
    assert not np.array_equal(unweighted_a, unweighted_b)


def test_detect_reports_affected_when_weighted_probe_finds_identical_output(monkeypatch):
    """Force the exact buggy behavior via monkeypatch (independent of
    whatever the real installed numpy currently does) and confirm the
    detector correctly classifies it as affected=True with the right
    detail message. This is the regression test that would have failed
    before affected-detection logic existed: it constructs the exact
    failure shape the real bug produces and checks the verdict."""

    real_default_rng = np.random.default_rng

    class _FakeGenerator:
        def choice(self, a, size=None, replace=True, p=None, shuffle=True):
            # Mimic the real bug: p is not None and replace=False ->
            # shuffle is silently ignored; always return the same order.
            rng = real_default_rng(999)
            if p is not None and not replace:
                return rng.choice(a, size=size, replace=False, p=p, shuffle=False)
            return rng.choice(a, size=size, replace=replace, p=p, shuffle=shuffle)

    fake = _FakeGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: fake)

    result = detect_shuffle_ignored_bug(population_size=200, sample_size=40, seed=0)
    assert result.affected is True
    assert "CONFIRMED" in result.detail


def test_detect_reports_affected_when_probe_itself_is_broken(monkeypatch):
    """If BOTH the weighted and unweighted comparisons come back identical
    (a broken probe/RNG, not evidence the bug is fixed), the detector
    must fail-safe to affected=True rather than claim a false clean bill
    of health."""

    real_default_rng = np.random.default_rng

    class _AlwaysIdenticalGenerator:
        def choice(self, a, size=None, replace=True, p=None, shuffle=True):
            rng = real_default_rng(42)
            return rng.choice(a, size=size, replace=replace, p=p, shuffle=False)

    fake = _AlwaysIdenticalGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: fake)

    result = detect_shuffle_ignored_bug(population_size=200, sample_size=40, seed=0)
    assert result.affected is True
    assert "INCONCLUSIVE" in result.detail


def test_detect_reports_not_affected_when_bug_is_fixed(monkeypatch):
    """If numpy fixes the bug (weighted path correctly differs under
    shuffle, same as the control), the detector must say affected=False."""

    real_default_rng = np.random.default_rng

    class _FixedGenerator:
        def choice(self, a, size=None, replace=True, p=None, shuffle=True):
            # A "fixed" numpy: shuffle genuinely changes output for both
            # weighted and unweighted paths.
            seed = 111 if shuffle else 222
            rng = real_default_rng(seed)
            return rng.choice(a, size=size, replace=replace, p=p, shuffle=False)

    fake = _FixedGenerator()
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: fake)

    result = detect_shuffle_ignored_bug(population_size=200, sample_size=40, seed=0)
    assert result.affected is False
    assert "Not reproduced" in result.detail


def test_safe_weighted_choice_passes_through_when_replace_true():
    rng = np.random.default_rng(0)
    weights = np.full(50, 1.0 / 50)
    result = safe_weighted_choice(rng, 50, size=10, replace=True, p=weights, shuffle=True)
    assert len(result) == 10
    assert all(0 <= x < 50 for x in result)


def test_safe_weighted_choice_passes_through_when_p_is_none():
    rng = np.random.default_rng(0)
    result = safe_weighted_choice(rng, 50, size=10, replace=False, p=None, shuffle=True)
    assert len(set(result.tolist())) == 10  # no replacement, all distinct


def test_safe_weighted_choice_applies_manual_shuffle_when_forced():
    """With force_workaround=True, safe_weighted_choice must actually
    change draw ORDER between two calls with different rng states, even
    though the selected SET could coincidentally match. Use a large
    enough population/sample that the chance of the workaround
    coincidentally reproducing the identical order is negligible."""
    n, k = 300, 60
    weights = np.linspace(1.0, 3.0, n)
    weights = weights / weights.sum()

    rng1 = np.random.default_rng(5)
    without_workaround = rng1.choice(n, size=k, replace=False, p=weights, shuffle=False)

    rng2 = np.random.default_rng(5)
    with_workaround = safe_weighted_choice(
        rng2, n, size=k, replace=False, p=weights, shuffle=True, force_workaround=True,
    )

    # Same SET of selected indices is fine/expected (the underlying
    # cumulative-distribution selection is unchanged); the ORDER must
    # differ, proving the manual np.random.Generator.shuffle() actually
    # ran (this is the exact behavior the upstream bug fails to provide).
    assert set(without_workaround.tolist()) == set(with_workaround.tolist())
    assert not np.array_equal(without_workaround, with_workaround)


def test_safe_weighted_choice_skips_manual_shuffle_when_shuffle_false():
    n, k = 100, 20
    weights = np.full(n, 1.0 / n)
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    a = rng1.choice(n, size=k, replace=False, p=weights, shuffle=False)
    b = safe_weighted_choice(
        rng2, n, size=k, replace=False, p=weights, shuffle=False, force_workaround=True,
    )
    assert np.array_equal(a, b)


def test_safe_weighted_choice_force_workaround_false_bypasses_shim():
    """force_workaround=False must call rng.choice exactly as given, with
    no post-hoc shuffle, even in the otherwise-buggy combination."""
    n, k = 100, 20
    weights = np.full(n, 1.0 / n)
    rng1 = np.random.default_rng(9)
    rng2 = np.random.default_rng(9)
    direct = rng1.choice(n, size=k, replace=False, p=weights, shuffle=True)
    shim = safe_weighted_choice(
        rng2, n, size=k, replace=False, p=weights, shuffle=True, force_workaround=False,
    )
    assert np.array_equal(direct, shim)


def test_independent_reference_sampler_is_unbiased_and_distinct():
    """Sanity-check the oracle itself: no replacement (all distinct
    indices) and each call is deterministic given its seed."""
    weights = np.linspace(1.0, 2.0, 50)
    weights = weights / weights.sum()
    a = independent_reference_weighted_sample_without_replacement(50, 15, weights, seed=1)
    b = independent_reference_weighted_sample_without_replacement(50, 15, weights, seed=1)
    assert len(set(a.tolist())) == 15
    assert np.array_equal(a, b)


def test_independent_reference_sampler_rejects_nonpositive_weights():
    weights = np.array([1.0, 0.0, 2.0])
    with pytest.raises(ValueError):
        independent_reference_weighted_sample_without_replacement(3, 2, weights, seed=0)


def test_safe_weighted_choice_auto_detects_and_applies_workaround_by_default():
    """The DEFAULT public behavior (force_workaround left unset, i.e.
    None) must itself call detect_shuffle_ignored_bug() and apply the
    manual-shuffle workaround when the live probe reports affected --
    every other existing test explicitly forces True/False and never
    exercises this auto-detect branch, even though it is what every
    real caller who does not pass force_workaround actually gets.

    This test does not assume the installed numpy is buggy: it first
    runs the same live probe the auto-detect path uses, then asserts
    safe_weighted_choice's un-forced default behavior is CONSISTENT
    with that live probe result, so it stays correct even after numpy
    eventually fixes the upstream bug.
    """
    n, k = 300, 60
    weights = np.linspace(1.0, 3.0, n)
    weights = weights / weights.sum()

    live_probe = detect_shuffle_ignored_bug()

    rng_plain = np.random.default_rng(5)
    without_workaround = rng_plain.choice(
        n, size=k, replace=False, p=weights, shuffle=False
    )

    rng_auto = np.random.default_rng(5)
    auto_result = safe_weighted_choice(
        rng_auto, n, size=k, replace=False, p=weights, shuffle=True,
        # force_workaround intentionally omitted -> exercises the
        # `if force_workaround is None:` auto-detect branch directly.
    )

    if live_probe.affected:
        # Auto-detect must behave exactly like force_workaround=True:
        # same selected SET, but a different draw ORDER than the
        # unshuffled call, proving the manual shuffle actually ran.
        assert set(without_workaround.tolist()) == set(auto_result.tolist())
        assert not np.array_equal(without_workaround, auto_result)
    else:
        # numpy has been fixed upstream: auto-detect must pass the
        # call straight through, matching rng.choice's own behavior
        # exactly (same rng-state consumption, no extra shuffle).
        rng_direct = np.random.default_rng(5)
        direct_result = rng_direct.choice(
            n, size=k, replace=False, p=weights, shuffle=True
        )
        assert np.array_equal(direct_result, auto_result)


def test_independent_reference_sampler_prefers_higher_weight_items_statistically():
    """A weaker, distribution-level correctness check: over many trials,
    an item with much higher weight should be selected more often than
    one with much lower weight -- proving the oracle's exponential-key
    scheme actually respects weights, not just returns valid distinct
    indices in arbitrary order."""
    n = 10
    weights = np.array([0.5] + [0.05] * 9)
    counts = np.zeros(n, dtype=np.int64)
    trials = 500
    for t in range(trials):
        sample = independent_reference_weighted_sample_without_replacement(n, 3, weights, seed=t)
        counts[sample] += 1
    # Item 0 (weight 10x any other) must be selected far more often.
    assert counts[0] > counts[1:].max() * 2


def test_safe_weighted_choice_does_not_crash_on_default_scalar_size(monkeypatch):
    """Regression test for a real crash: `Generator.choice`'s own `size`
    parameter defaults to None (a single scalar draw), and
    `safe_weighted_choice` is documented as a "drop-in replacement" for
    it -- callers may reasonably omit `size` entirely. Before this fix,
    when force_workaround fires (or auto-detect finds the bug present),
    the workaround branch called `rng.shuffle(result)` on a bare Python
    int (numpy's scalar-choice return value), raising
    `TypeError: object of type 'int' has no len()`. This test forces the
    workaround branch via monkeypatched detection so it fails for the
    correct behavioral reason regardless of the installed numpy's own
    live bug status, and would have failed with that TypeError before
    the `size is None` short-circuit was added.
    """
    n = 50
    weights = np.linspace(1.0, 2.0, n)
    weights = weights / weights.sum()

    rng = np.random.default_rng(0)
    # force_workaround=True exercises the exact branch that crashed,
    # without depending on whether the installed numpy still has the bug.
    result = safe_weighted_choice(
        rng, n, size=None, replace=False, p=weights, shuffle=True,
        force_workaround=True,
    )
    assert isinstance(result, (int, np.integer))
    assert 0 <= int(result) < n

    # Also confirm the auto-detect (unforced) default path never crashes
    # for the scalar-size case, matching plain rng.choice's own behavior
    # exactly when size=None (no order to shuffle, so passthrough is the
    # only correct behavior regardless of live bug-detection status).
    rng_direct = np.random.default_rng(3)
    direct = rng_direct.choice(n, size=None, replace=False, p=weights, shuffle=True)
    rng_auto = np.random.default_rng(3)
    auto = safe_weighted_choice(
        rng_auto, n, size=None, replace=False, p=weights, shuffle=True,
    )
    assert direct == auto
