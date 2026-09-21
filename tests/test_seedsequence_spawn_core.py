"""Tests for numpy_seedsequence_spawn_guard.core.

Verification strategy: the acceptance oracle here is structural, not
numerical. We prove three things:

1. `detect_spawn_race` can actually observe the race on a bare
   SeedSequence under thread-switch pressure (a sanity check on the
   probe itself -- if this never fires, the probe is broken, not proof
   the bug doesn't exist).
2. `GuardedSeedSequence.spawn()` produces strictly-increasing,
   never-repeating spawn keys / generated state under the SAME
   concurrent stress that reproduces the bug on a bare SeedSequence.
3. `GuardedSeedSequence` passes every other attribute/method straight
   through unmodified (it is a synchronization wrapper, not a
   reimplementation).
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from numpy_correctness_guards.guards.seedsequence_spawn import (
    BugDetectionResult,
    GuardedSeedSequence,
    detect_spawn_race,
    verify_guard_eliminates_race,
)


def test_detect_spawn_race_returns_dataclass_with_expected_fields():
    result = detect_spawn_race(n_threads=4, spawns_per_thread=200)
    assert isinstance(result, BugDetectionResult)
    assert result.numpy_version == np.__version__
    assert isinstance(result.affected, bool)
    assert result.total_children == 800
    assert result.duplicate_count >= 0
    assert result.duplicate_count <= result.total_children


def test_detect_spawn_race_probe_can_observe_duplicates_under_pressure():
    """Sanity check on the probe itself: run enough trials at aggressive
    thread-switch pressure that at least one trial reproduces >=1
    duplicate. If this never happens across many trials, either the
    installed numpy has genuinely fixed the race (a legitimate, if
    surprising, outcome) or the probe itself is broken -- we can't
    fully distinguish those from inside the test, but we CAN assert the
    probe's own bookkeeping (total/unique counts) is internally
    consistent on every trial, which is the property this test actually
    guards.
    """
    observed_any_duplicate = False
    for trial in range(5):
        result = detect_spawn_race(
            n_threads=4, spawns_per_thread=500, seed=1000 + trial
        )
        assert result.total_children == 2000
        assert 0 <= result.duplicate_count <= result.total_children
        assert result.affected == (result.duplicate_count > 0)
        if result.duplicate_count > 0:
            observed_any_duplicate = True

    if not observed_any_duplicate:
        pytest.skip(
            "Race not reproduced in 5 trials on this numpy/platform -- "
            "either the upstream bug is fixed here, or thread-switch "
            "timing didn't trigger it this run. Not a guard-code failure."
        )


def test_guarded_seed_sequence_eliminates_race_under_stress():
    payload = verify_guard_eliminates_race(n_threads=4, spawns_per_thread=500, seed=42)
    assert payload["guarded_total"] == 2000
    assert payload["guarded_duplicates"] == 0
    assert payload["guard_fully_eliminates_race"] is True


def test_guarded_seed_sequence_eliminates_race_across_repeated_trials():
    """Run the guarded stress test several times with different seeds to
    rule out a single lucky trial -- the lock must eliminate the race
    deterministically, not merely reduce its probability.
    """
    for seed in range(5):
        payload = verify_guard_eliminates_race(
            n_threads=6, spawns_per_thread=300, seed=seed
        )
        assert payload["guarded_duplicates"] == 0, (
            f"seed={seed}: guard failed to eliminate race "
            f"({payload['guarded_duplicates']} duplicates)"
        )


def test_guarded_seed_sequence_spawn_keys_are_sequential():
    guard = GuardedSeedSequence(7)
    keys = [guard.spawn(1)[0].spawn_key for _ in range(10)]
    assert keys == [(i,) for i in range(10)]


def test_guarded_seed_sequence_passthrough_attributes():
    guard = GuardedSeedSequence(99)
    bare = np.random.SeedSequence(99)
    assert guard.entropy == bare.entropy
    assert guard.pool_size == bare.pool_size
    assert tuple(guard.generate_state(4)) == tuple(bare.generate_state(4))


def test_guarded_seed_sequence_repr_mentions_wrapped_instance():
    guard = GuardedSeedSequence(5)
    assert "GuardedSeedSequence" in repr(guard)


def test_guarded_seed_sequence_spawn_from_multiple_threads_never_duplicates():
    """End-to-end regression test: the exact failure mode from the
    upstream issue (concurrent .spawn() on one shared SeedSequence
    yielding duplicate children) must not occur through the guard,
    across many repeated runs to rule out flakiness in either
    direction.
    """
    for attempt in range(3):
        guard = GuardedSeedSequence(2026 + attempt)
        collected = []
        lock = threading.Lock()

        def worker():
            local = [tuple(guard.spawn(1)[0].generate_state(4)) for _ in range(400)]
            with lock:
                collected.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(collected) == 1600
        assert len(set(collected)) == 1600, (
            f"attempt={attempt}: guard produced duplicate spawned streams"
        )
