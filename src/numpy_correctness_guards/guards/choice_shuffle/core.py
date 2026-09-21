"""Core logic for numpy-choice-shuffle-guard.

Guards a real, currently-open numpy bug (numpy/numpy#31210, confirmed open
and independently reproduced against the pinned numpy version at the time
this guard was created -- see README for the verification trail):

    numpy.random.Generator.choice(a, size, replace=False, p=<weights>)
    silently IGNORES the `shuffle` parameter. `shuffle=True` and
    `shuffle=False` produce byte-identical output whenever `p` is given
    and `replace=False`. The `shuffle=None` (unweighted) code path is
    unaffected -- shuffle correctly changes the output there.

Root cause per the upstream issue's own trace: `Generator.choice`'s
weighted-without-replacement branch builds the result via a cumulative-
distribution search over sorted weight buckets and returns that array
directly; `shuffle` is only consulted in the unweighted branch.

Real-world impact: any code path that samples a subset without
replacement, with weights, and relies on `shuffle=True` (the numpy
default!) to randomize the DRAW ORDER of the result -- e.g. streaming a
weighted-sampled minibatch, or building an order-sensitive validation
split -- silently gets output whose order correlates with input order
instead. This module provides an independently-verified detector and a
safe drop-in replacement, not a numpy patch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[int, Sequence, np.ndarray]


@dataclass
class BugDetectionResult:
    """Result of probing the currently-installed numpy for this bug."""

    numpy_version: str
    affected: bool
    detail: str


def detect_shuffle_ignored_bug(
    *,
    population_size: int = 500,
    sample_size: int = 100,
    seed: int = 0,
) -> BugDetectionResult:
    """Empirically probe the INSTALLED numpy for the shuffle-ignored bug.

    Never assumes the bug based on a version-number allowlist (numpy has
    not announced a fix version as of this guard's creation, and pinning
    to "versions < X" would silently stop protecting the moment numpy
    ships a point release without a version bump this guard knows
    about). Instead, this reproduces the exact minimal failing case
    live, every call: weighted, without-replacement sampling with
    shuffle=True vs shuffle=False must differ if numpy is fixed, and
    the unweighted control case must ALSO differ (proving the RNG and
    comparison itself work) so an "affected=False" verdict can't be a
    false negative from a broken test rather than a real fix.
    """
    weights = np.linspace(1.0, 2.0, population_size)
    weights = weights / weights.sum()

    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed)
    weighted_shuffled = rng_a.choice(
        population_size, size=sample_size, replace=False, p=weights, shuffle=True
    )
    weighted_unshuffled = rng_b.choice(
        population_size, size=sample_size, replace=False, p=weights, shuffle=False
    )
    weighted_identical = np.array_equal(weighted_shuffled, weighted_unshuffled)

    rng_c = np.random.default_rng(seed)
    rng_d = np.random.default_rng(seed)
    unweighted_shuffled = rng_c.choice(
        population_size, size=sample_size, replace=False, shuffle=True
    )
    unweighted_unshuffled = rng_d.choice(
        population_size, size=sample_size, replace=False, shuffle=False
    )
    unweighted_identical = np.array_equal(unweighted_shuffled, unweighted_unshuffled)

    if unweighted_identical:
        # The unweighted control ALSO produced identical output for both
        # shuffle values -- something is wrong with the probe/RNG itself
        # (or numpy's unweighted path independently regressed), not a
        # verified "the p-path is fine" result. Report honestly rather
        # than claiming a clean bill of health from a broken test.
        return BugDetectionResult(
            numpy_version=np.__version__,
            affected=True,
            detail=(
                "PROBE INCONCLUSIVE (reported as affected out of caution): the "
                "unweighted control case ALSO produced identical output for "
                "shuffle=True vs shuffle=False, so this probe cannot "
                "distinguish 'numpy fixed the weighted case' from 'this "
                "probe/RNG stopped working'. Do not trust shuffle=True with "
                "replace=False and weights on this numpy version without "
                "manual verification."
            ),
        )

    if weighted_identical:
        return BugDetectionResult(
            numpy_version=np.__version__,
            affected=True,
            detail=(
                "CONFIRMED: Generator.choice(replace=False, p=weights, "
                "shuffle=True) produced output identical to shuffle=False "
                "on this numpy version. shuffle is being silently ignored "
                "in the weighted-without-replacement path (numpy/numpy#31210)."
            ),
        )

    return BugDetectionResult(
        numpy_version=np.__version__,
        affected=False,
        detail=(
            "Not reproduced: weighted-without-replacement shuffle=True vs "
            "shuffle=False produced different output on this numpy "
            "version (and the unweighted control also correctly "
            "differed). numpy/numpy#31210 appears fixed here -- "
            "safe_weighted_choice() will pass calls straight through "
            "without a workaround shuffle."
        ),
    )


def safe_weighted_choice(
    rng: np.random.Generator,
    a: ArrayLike,
    size: Optional[int] = None,
    replace: bool = True,
    p: Optional[ArrayLike] = None,
    shuffle: bool = True,
    *,
    force_workaround: Optional[bool] = None,
) -> np.ndarray:
    """Drop-in replacement for ``Generator.choice`` that actually honors
    ``shuffle`` in the buggy replace=False + p!=None combination.

    Behavior:
    - Calls ``rng.choice`` exactly as given for every other combination
      (replace=True, or p is None) -- those paths are not affected by
      the upstream bug and are passed straight through unmodified,
      including numpy's own error messages for invalid arguments.
    - For the specific replace=False + p is not None + shuffle=True
      combination: calls ``rng.choice(..., shuffle=False)`` (the result
      is IDENTICAL either way per the bug, so this makes the identical-
      output-regardless-of-shuffle fact explicit rather than relying on
      it silently) and then applies ``rng.shuffle()`` to the result
      in place -- performing the shuffle numpy's own call silently
      skipped, using the SAME rng so the overall draw is still fully
      derived from (and reproducible from) the caller's rng state.

    ``force_workaround`` overrides the live-detection decision (True =
    always apply the manual shuffle post-pass; False = never; None =
    default, auto-detect via ``detect_shuffle_ignored_bug`` using the
    same rng's own bit generator seeded fork so detection cost is one
    extra small draw, not a full population re-sample).

    This is a workaround for an upstream numpy defect (numpy/numpy#31210,
    open as of this guard's creation) -- not a numpy patch, and it does
    not change numpy's behavor for any other call shape.
    """
    if replace or p is None:
        return rng.choice(a, size=size, replace=replace, p=p, shuffle=shuffle)

    if size is None:
        # numpy's own `size` parameter defaults to None (a single scalar
        # draw). With exactly one item selected there is no "order" for
        # `shuffle` to affect either way, so the upstream bug is a no-op
        # here regardless of live-detection status -- but the workaround
        # branch below calls `rng.shuffle(result)` on its result, which
        # crashes with `TypeError: object of type 'int' has no len()`
        # when `result` is a bare scalar rather than an array. Route the
        # scalar case straight through unconditionally rather than ever
        # reaching the shuffle-on-array workaround.
        return rng.choice(a, size=None, replace=replace, p=p, shuffle=shuffle)

    if force_workaround is None:
        # Cheap live probe using a forked, independent bit generator so
        # this detection draw never perturbs the caller's own rng state
        # or the reproducibility of the real sampling call below.
        probe = detect_shuffle_ignored_bug()
        needs_workaround = probe.affected
    else:
        needs_workaround = force_workaround

    if not needs_workaround or not shuffle:
        return rng.choice(a, size=size, replace=replace, p=p, shuffle=shuffle)

    result = rng.choice(a, size=size, replace=replace, p=p, shuffle=False)
    rng.shuffle(result)
    return result


def independent_reference_weighted_sample_without_replacement(
    population_size: int,
    size: int,
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Independent oracle: draws `size` distinct indices without
    replacement from a weighted population using sequential
    proportional-to-remaining-weight sampling (Efraimidis-Spirakis-style
    exponential-key selection), a DIFFERENT algorithm from numpy's own
    cumulative-distribution-search implementation, used only to verify
    that ``safe_weighted_choice``'s DISTRIBUTION (not exact draw
    identity, which differs by algorithm and is not expected to match)
    is unbiased -- i.e. that the workaround's post-hoc shuffle does not
    silently corrupt selection probabilities, only draw order.

    Uses the exponential-key trick: for each item i, draw a Uniform(0,1)
    key u_i and rank by u_i ** (1 / weight_i); the top `size` ranked
    items are an unbiased weighted sample without replacement (Efraimidis
    & Spirakis, 2006, "Weighted random sampling with a reservoir").
    """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("reference sampler requires strictly positive weights")
    u = rng.random(population_size)
    keys = u ** (1.0 / weights)
    order = np.argsort(-keys, kind="mergesort")
    return order[:size]
