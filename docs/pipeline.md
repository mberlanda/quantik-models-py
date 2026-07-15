# Pipeline

This repository owns the model-training side of the Quantik pipeline. The
contracts repository remains the source of truth for artifact IDs and field
semantics.

## One-command smoke

```bash
export CONTRACTS=/Users/mauroberlanda/Code/quantik-ns/quantik-core-contracts
export RUST=/Users/mauroberlanda/Code/quantik-ns/quantik-core-rust
export CORE_PY=/Users/mauroberlanda/Code/quantik-ns/quantik/quantik-core-py
export MODELS=/Users/mauroberlanda/Code/quantik-ns/quantik-models-py
cd "$MODELS"
scripts/run_smoke_pipeline.sh
```

The script runs:

1. contract validation,
2. Rust depth-6 opening-book generation,
3. Rust position generation using the book for exact references,
4. Rust observation generation across engines,
5. Rust H2H generation and report rendering,
6. Rust MCTS self-play export,
7. Python materialization into `.npz` training views.

Large runs should override counts with environment variables such as
`OPENING_POSITIONS`, `MCTS_ITERATIONS`, `SELFPLAY_GAMES`, and `OUT`.
