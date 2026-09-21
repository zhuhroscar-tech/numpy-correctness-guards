# numpy-correctness-guards

Consolidated NumPy correctness guards: small, reproducible probes for real NumPy edge-case bugs plus safe workaround helpers where appropriate.

This package replaces separate one-bug repositories with one shared CLI/API surface.

## Included guards

- `choice-shuffle` — detects and works around `numpy.random.Generator.choice(..., replace=False, p=weights, shuffle=True)` ignoring `shuffle` in the weighted-without-replacement path (`numpy/numpy#31210`). Migrated from `numpy-choice-shuffle-guard`.

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
```

The legacy executable name is intentionally not preserved here; use the shared `numpy-guard` entry point for all consolidated guards.

## Python API

```python
import numpy as np
from numpy_correctness_guards.guards.choice_shuffle import safe_weighted_choice

rng = np.random.default_rng(0)
weights = np.linspace(1.0, 2.0, 100)
weights = weights / weights.sum()
result = safe_weighted_choice(rng, 100, size=20, replace=False, p=weights)
```

## License

MIT.
