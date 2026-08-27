# Why some tests skip without torch

`.github/workflows/e2e-data-pipeline.yml` installs `quantik-models[dev,arrow]`
— **no torch**. That is deliberate: `data/dataset.py` states the base install
must be able to inspect datasets without an accelerator stack, and the E2E
pipeline exercises exactly that path.

Two rules follow, and PR #4 broke both before this was noticed:

1. **Torch-dependent tests guard themselves.** Use
   `pytest.importorskip("torch")` — module-level if the whole file needs it,
   inside the test if only one case does. This is the convention already used
   by `test_trainer.py` and `test_export_checkpoint.py`.
2. **Pure-NumPy helpers do not live behind a torch import.** `split_by_key`,
   `ply_sampling_weights` and the weighted metric merge were originally inside
   `train/supervised.py`, which imports torch at module scope — so testing them
   required torch they did not need. They now live in `data/exact_corpus.py`
   and `train/metrics.py`; `supervised.py` imports them, so its surface is
   unchanged.

## Reproducing the CI environment locally

Rather than guessing, block the modules the way a missing install does:

```bash
mkdir -p /tmp/no_torch && cat > /tmp/no_torch/sitecustomize.py <<'PY'
import sys
BLOCKED = {"torch", "safetensors"}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}", name=name)
        return None
sys.meta_path.insert(0, _Blocker())
PY

PYTHONPATH=/tmp/no_torch .venv/bin/python -m pytest -q   # expect: 80 passed, 4 skipped
.venv/bin/python -m pytest -q                            # expect: 91 passed
```

A shim that raises plain `ImportError` is **not** a faithful simulation —
`pytest.importorskip` treats a module that exists but fails to import as a real
error and re-raises, so the run errors during collection instead of skipping.
A genuinely absent module raises `ModuleNotFoundError`.
