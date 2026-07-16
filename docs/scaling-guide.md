# Scaling From Smoke To The Target Model

The trainer exposes one architecture (`PolicyValueNet`) at three named
presets. Scaling is a one-flag change; what actually needs to grow with
the model is the data.

## Presets

| Preset | Channels | Blocks | Parameters | float32 safetensors | Intended use |
| --- | --- | --- | --- | --- | --- |
| `smoke` | 16 | 2 | 13,991 | ~0.1 MiB | CI, examples, plumbing checks |
| `small` | 64 | 4 | 304,711 | ~1.2 MiB | Laptop baselines, ablations |
| `target` | 256 | 13 | 15,374,023 | ~58.6 MiB | The 50-100 MB contract model |

(Exact parameter counts are asserted by `tests/test_policy_value_net.py`;
the `target` preset must land inside the 50-100 MB envelope from
quantik-core-contracts `docs/policy-value-model-project.md`.)

## What changes when you scale

Only the preset flag:

    quantik-models-train --npz ... --preset small
    quantik-models-train --npz ... --preset target

Everything else (loss, masking, sharding, export, manifest) is
identical at every scale. `--channels/--blocks` override the presets
for ablations.

## What must grow: data

| Preset | Minimum sensible corpus |
| --- | --- |
| `smoke` | the tiny pipeline corpus (tens of rows) — plumbing only |
| `small` | >= 100k rows: full observation runs across MCTS/minimax/beam plus self-play (see the data milestones in the contracts model project doc) |
| `target` | the full depth-6 book corpus: book-backed positions, multi-engine observations, large self-play, autoplay rounds |

Quantik's reachable state space is small compared to `target`'s 15.4M
parameters. Training `target` on a small corpus will memorize it
perfectly and generalize poorly; the paper
(`docs/policy-value-training-paper.md`) discusses why the envelope is
still useful (headroom for regularization, distillation targets, and
future feature channels) and when a smaller deployed model is the
better trade.

## Wall-clock expectations

Measured on the smoke corpus (tens of rows), batch 16, per epoch:
`smoke` sub-second on CPU; `small` ~1-2 s CPU, sub-second MPS;
`target` minutes on CPU — use MPS/CUDA (`--device auto` picks them up)
and a real corpus.

## Verifying the envelope

    ls -l <out-dir>/weights.safetensors     # 50-100 MB for target
    python - <<'PY'
    import json; m = json.load(open("<out-dir>/manifest.json"))
    print(m["size_bytes"] / 2**20, "MiB", m["parameter_count"], "params")
    PY

The exported manifest must validate through quantik-core-py
(`load_model_checkpoint_manifest`) — the export test does this on every
CI run.

## Acceptance gates beyond size

A checkpoint is only useful if it clears the gates in the contracts
model project doc: manifest validation, fixed-budget search
improvement, book-frontier H2H, near-zero illegal mass after masking,
reproducible training data. Size is the cheapest gate; H2H is the one
that matters.
