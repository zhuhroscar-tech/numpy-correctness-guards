# numpy-correctness-guards

Consolidated NumPy correctness guards: small, reproducible probes for real NumPy edge-case bugs plus safe workaround helpers where appropriate.

This package replaces separate one-bug repositories with one shared CLI/API surface.

## Included guards

- `choice-shuffle` — detects and works around `numpy.random.Generator.choice(..., replace=False, p=weights, shuffle=True)` ignoring `shuffle` in the weighted-without-replacement path (`numpy/numpy#31210`). Migrated from `numpy-choice-shuffle-guard`.
- `einsum-newdtype` — detects and works around `np.einsum` returning wrong results or crashing for new-style dtype operands (`numpy/numpy#32671`). Migrated from `numpy-einsum-newdtype-guard`.
- `poisson-variance` — detects and works around `numpy.random.Generator.poisson(lam)` inflating variance for very large `lam` values (`numpy/numpy#31986`). Migrated from `numpy-poisson-variance-guard`.
- `seedsequence-spawn` — detects and works around `numpy.random.SeedSequence.spawn()` handing out duplicate child streams under concurrent calls on one shared parent (`numpy/numpy#32063`). Migrated from `numpy-seedsequence-spawn-guard`.
- `timedelta64-floordiv` — detects and works around `numpy.timedelta64(...) // int` truncating negative inexact divisions toward zero instead of flooring (`numpy/numpy#32522`). Migrated from `numpy-timedelta64-floordiv-guard`.

## Install

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

## CLI

List available guards:

```bash
numpy-guard list
```

Run the migrated choice/shuffle detector:

```bash
numpy-guard run choice-shuffle detect --json
numpy-guard run choice-shuffle verify --trials 500 --tolerance 0.08
numpy-guard run einsum-newdtype detect --json
numpy-guard run poisson-variance detect --lam 1e16 --samples 200000 --json
numpy-guard run poisson-variance sample --lam 1e16 --size 1000 --seed 42
numpy-guard run seedsequence-spawn detect --json
numpy-guard run seedsequence-spawn verify --threads 4 --spawns-per-thread 500
numpy-guard run timedelta64-floordiv detect --json
numpy-guard run timedelta64-floordiv verify --json
numpy-guard run timedelta64-floordiv apply --value -7 --unit us --divisor 2
```

The legacy executable name is intentionally not preserved here; use the shared `numpy-guard` entry point for all consolidated guards.

## Python API

```python
import numpy as np
from numpy_correctness_guards.guards.choice_shuffle import safe_weighted_choice
from numpy_correctness_guards.guards.einsum_newdtype import safe_einsum
from numpy_correctness_guards.guards.poisson_variance import safe_poisson
from numpy_correctness_guards.guards.seedsequence_spawn import GuardedSeedSequence
from numpy_correctness_guards.guards.timedelta64_floordiv import safe_timedelta64_floordiv

rng = np.random.default_rng(0)
weights = np.linspace(1.0, 2.0, 100)
weights = weights / weights.sum()
result = safe_weighted_choice(rng, 100, size=20, replace=False, p=weights)
matrix = safe_einsum("ij,jk->ik", np.eye(2), np.eye(2))
samples = safe_poisson(1e16, 1000, rng=rng)
shared = GuardedSeedSequence(12345)
children = shared.spawn(4)
bucket = safe_timedelta64_floordiv(np.timedelta64(-7, "us"), 2)
```

The `einsum-newdtype` live probe needs the optional `numpy_quaddtype` package, which can be installed with:

```bash
python -m pip install '.[quaddtype]'
```

## License

MIT.
