"""Core logic for numpy-seedsequence-spawn-guard.

Guards a real, currently-open numpy bug (numpy/numpy#32063, confirmed open
and independently reproduced against the pinned numpy version at the time
this guard was created -- see README for the verification trail):

    numpy.random.SeedSequence.spawn() is NOT thread-safe. It reads
    self.n_children_spawned, does Python-level work (np.concatenate
    inside get_assembled_entropy) that can release the GIL, and only
    afterwards increments the counter. Two threads calling .spawn() on
    the SAME SeedSequence instance concurrently can both observe the
    same starting index and hand back children with IDENTICAL spawn
    keys and therefore IDENTICAL derived random streams -- silently
    breaking the independent-streams guarantee spawning exists to
    provide. There is no error, warning, or exception; the corruption
    is silent and only shows up as suspiciously-correlated "independent"
    samples downstream (e.g. in a multi-worker DataLoader or a
    multiprocessing/threading fan-out that shares one parent
    SeedSequence).

Real-world impact: any code path that shares ONE SeedSequence across
multiple threads and calls `.spawn()` concurrently to hand each thread
an "independent" stream -- a common pattern for parallel data loading,
Monte Carlo workers, or ensemble training -- can silently receive
duplicate streams for some subset of threads, corrupting downstream
statistics without any visible error.

This module provides an independently-verified detector (a live race
probe, not a version-number guess) and a safe drop-in wrapper that
serializes .spawn() calls on a shared SeedSequence with a lock -- not a
numpy patch.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BugDetectionResult:
    """Result of probing the installed numpy for the spawn thread-safety bug."""

    numpy_version: str
    affected: bool
    detail: str
    duplicate_count: int
    total_children: int


def _spawn_stress(
    seed_sequence: np.random.SeedSequence,
    *,
    n_threads: int,
    spawns_per_thread: int,
    use_lock: bool,
) -> List[tuple]:
    """Spawn `n_threads * spawns_per_thread` children from the SAME
    seed_sequence concurrently, optionally serializing with a lock, and
    return the raw generated-state tuples (not just spawn_key -- a
    duplicate spawn_key always implies a duplicate stream, but comparing
    the actual generated state is the more direct, assumption-free
    signal of "these two children would produce identical random
    output").
    """
    collected: List[tuple] = []
    collect_lock = threading.Lock()
    spawn_lock = threading.Lock() if use_lock else None

    def worker() -> None:
        local = []
        for _ in range(spawns_per_thread):
            if spawn_lock is not None:
                with spawn_lock:
                    child = seed_sequence.spawn(1)[0]
            else:
                child = seed_sequence.spawn(1)[0]
            local.append(tuple(child.generate_state(4)))
        with collect_lock:
            collected.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return collected


def detect_spawn_race(
    *,
    n_threads: int = 4,
    spawns_per_thread: int = 500,
    seed: int = 12345,
    switch_interval: float = 1e-5,
) -> BugDetectionResult:
    """Empirically probe the INSTALLED numpy for the SeedSequence.spawn
    thread-safety race.

    Never assumes the bug from a version-number allowlist (numpy has not
    announced a fix version as of this guard's creation). Instead this
    reproduces the exact race live, every call: many threads spawn from
    one shared SeedSequence with the interpreter's thread-switch interval
    forced very short (to maximize the chance of hitting the race window
    inside a single run rather than needing a long soak). If ANY of the
    spawned children's generated state collides, the race is present --
    a truly race-free spawn() would produce zero collisions across
    thousands of draws, since spawn_key is meant to be a strictly
    increasing per-call counter.

    A short-switch-interval stress test is inherently probabilistic: a
    fixed numpy could theoretically still pass by chance if this probe
    found zero duplicates, so `affected=False` here means "no race
    observed in this run", not a mathematical proof of safety. Reports
    the duplicate count honestly either way rather than collapsing to a
    boolean the caller can't audit.
    """
    sys.setswitchinterval(switch_interval)
    try:
        ss = np.random.SeedSequence(seed)
        children = _spawn_stress(
            ss, n_threads=n_threads, spawns_per_thread=spawns_per_thread, use_lock=False
        )
    finally:
        sys.setswitchinterval(0.005)  # restore Python's default

    total = len(children)
    unique = len(set(children))
    duplicates = total - unique

    if duplicates > 0:
        return BugDetectionResult(
            numpy_version=np.__version__,
            affected=True,
            detail=(
                f"CONFIRMED: {duplicates} of {total} concurrently-spawned "
                "children from one shared SeedSequence produced DUPLICATE "
                "generated state under concurrent .spawn() calls "
                "(numpy/numpy#32063). Independent-streams guarantee is "
                "broken for unsynchronized concurrent spawning on this "
                "numpy version."
            ),
            duplicate_count=duplicates,
            total_children=total,
        )

    return BugDetectionResult(
        numpy_version=np.__version__,
        affected=False,
        detail=(
            f"Not reproduced in this run: all {total} concurrently-spawned "
            "children had distinct generated state. This does not prove "
            "the race is fixed (a probabilistic race can pass by chance) "
            "-- guarded_spawn() is still recommended for any code sharing "
            "a SeedSequence across threads."
        ),
        duplicate_count=0,
        total_children=total,
    )


class GuardedSeedSequence:
    """Thread-safe wrapper around `numpy.random.SeedSequence` that
    serializes `.spawn()` calls with a lock, eliminating the race in
    numpy/numpy#32063 regardless of whether the installed numpy has
    fixed it.

    Every other attribute/method (entropy, spawn_key, generate_state,
    pool, state, n_children_spawned) passes straight through to the
    wrapped SeedSequence unmodified -- this is a synchronization
    wrapper, not a reimplementation. Only `.spawn()` is intercepted.

    Usage:
        shared = GuardedSeedSequence(12345)
        # safe to call `.spawn(1)` on `shared` from multiple threads
    """

    def __init__(self, *args, **kwargs) -> None:
        self._seed_sequence = np.random.SeedSequence(*args, **kwargs)
        self._lock = threading.Lock()

    def spawn(self, n_children: int):
        with self._lock:
            return self._seed_sequence.spawn(n_children)

    def __getattr__(self, name):
        # Only reached for attributes not found on GuardedSeedSequence
        # itself (spawn, __init__, _seed_sequence, _lock) -- delegates
        # everything else (entropy, spawn_key, generate_state, pool,
        # state, n_children_spawned) to the wrapped instance.
        return getattr(self._seed_sequence, name)

    def __repr__(self) -> str:
        return f"GuardedSeedSequence({self._seed_sequence!r})"


def verify_guard_eliminates_race(
    *,
    n_threads: int = 4,
    spawns_per_thread: int = 500,
    seed: int = 12345,
    switch_interval: float = 1e-5,
) -> dict:
    """Run the SAME stress test twice under the SAME forced thread-switch
    pressure: once against a bare SeedSequence (expected to show
    duplicates when the bug is present) and once through
    GuardedSeedSequence (expected to show ZERO duplicates always,
    proving the lock actually serializes access rather than merely
    reducing -- but not eliminating -- the race window).

    This is the acceptance oracle: a guard that only reduces contention
    without fully serializing would still show occasional duplicates at
    high thread counts, so this check is a real correctness proof for
    the wrapper, not a smoke test.
    """
    sys.setswitchinterval(switch_interval)
    try:
        bare = np.random.SeedSequence(seed)
        bare_children = _spawn_stress(
            bare, n_threads=n_threads, spawns_per_thread=spawns_per_thread, use_lock=False
        )

        guarded = GuardedSeedSequence(seed)
        guarded_lock = threading.Lock()
        collected: List[tuple] = []

        def worker() -> None:
            local = []
            for _ in range(spawns_per_thread):
                child = guarded.spawn(1)[0]
                local.append(tuple(child.generate_state(4)))
            with guarded_lock:
                collected.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(0.005)

    bare_total = len(bare_children)
    bare_unique = len(set(bare_children))
    guarded_total = len(collected)
    guarded_unique = len(set(collected))

    return {
        "bare_total": bare_total,
        "bare_unique": bare_unique,
        "bare_duplicates": bare_total - bare_unique,
        "guarded_total": guarded_total,
        "guarded_unique": guarded_unique,
        "guarded_duplicates": guarded_total - guarded_unique,
        "guard_fully_eliminates_race": (guarded_total - guarded_unique) == 0,
    }
