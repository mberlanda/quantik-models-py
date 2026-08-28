# smoke-best — committed checkpoint fixture

A real `model-checkpoint.v1` checkpoint, small enough to keep in version
control: `resnet-c16-b2`, 13,991 parameters, 59 KB of float32 safetensors.
It is the `smoke` preset trained on the tiny smoke corpus.

It is committed so that tests have a genuine checkpoint — real weights, a real
manifest, a real `weights_hash` — without a network fetch or a training run.
`runs/` is gitignored and always will be; training output does not belong in
the repository. This directory is the deliberate exception, and the size is the
reason it can be one. The forthcoming Rust parity test in `quantik-api-rust`,
which must assert that candle and PyTorch agree on the same logits, needs
exactly this: a fixed checkpoint both runtimes can load offline.

**This is not a useful player.** At 13,991 parameters trained on tens of rows
it exists to exercise plumbing — export, manifest validation, weight
round-tripping, tensor shapes. Judge nothing about model strength by it.

The checkpoints worth playing against are the `resnet-c128-b6` runs at
1,786,823 parameters (6.8 MB each), which are too large to commit on every
retrain. Those will be published as release artifacts, carrying the same three
files this directory holds; see `quantik-api-rust/docs/model-serving.md` for the
distribution plan and how the API will resolve a `model_id` to a checkpoint.
