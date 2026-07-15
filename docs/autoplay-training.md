# Autoplay Training Design

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
