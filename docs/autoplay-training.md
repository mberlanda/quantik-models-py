# Autoplay Training Design

> **About the numbers in this document.** Every measurement here comes from
> a specific run of a specific checkpoint, and the checkpoints live under
> `runs/`, which is gitignored — so nothing here can be verified from a
> fresh clone alone. Two things are worth checking before trusting a figure:
> **which learning rate it was measured at**, and **when**. Anything dated
> before 2026-08-30 was measured at `--lr 2e-3`, a rate chosen for the
> ResNet and inherited by every architecture added later; two of the four
> were being trained at the wrong one, and correcting that reversed several
> conclusions rather than merely shifting decimals. See
> `learning-rate-sweep.md`. Regenerate everything with
> `scripts/evaluate_lineup.sh`.


Autoplay should let engines discover stronger play while staying grounded in
contracts.

## Phase 1: supervised bootstrap

Use depth-6 opening-book references, search observations, and MCTS self-play to
train the first policy/value model.

## Phase 2: book-guided self-play

Start games with opening-book guidance up to depth 6, then fall back to a mix of
MCTS, beam, minimax, and model-guided search. Keep the opening book as a data
source, not a model dependency bundled into core.

Needed Rust work:

- self-play runner supporting `--book PATH`, `--opening-policy`, and engine
  pairs,
- position/frontier sampling from the book,
- exported `selfplay.v1` rows with provenance tags or companion manifests.

## Phase 3: active learning

Run model-guided engines against baseline engines. Export positions where the
model and search disagree, then label them with stronger search or exact book
lookups.
