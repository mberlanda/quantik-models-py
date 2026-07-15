# Frontend Play And Autoplay Integration

The frontend should reuse the same engine/artifact boundaries as model training.
Human play, CPU play, and autoplay should all flow through the core rules and
engine APIs rather than duplicating game logic in the UI.

## Modes

The first playable surface should support:

| Mode | Purpose | Engine dependency |
| --- | --- | --- |
| Human vs human | Validate board UX, move legality, history, undo/reset, and shared rules semantics. | Core rules only. |
| Human vs CPU | Let a person play against random/minimax/MCTS/beam and later model-guided engines. | Core rules plus one engine adapter. |
| CPU vs CPU autoplay | Visualize engine behavior and collect lightweight demos. | Engine adapters plus optional artifact export. |
| Book-guided CPU/autoplay | Start from opening-book recommendations up to a chosen depth, then fall back to engines. | Opening-book probe plus engine adapters. |

## Boundary

```text
frontend UI
  -> game session API
  -> quantik-core-py or quantik-core-rust rules engine
  -> engine adapters: random, minimax, MCTS, beam, future model-guided
  -> optional artifacts: selfplay.v1, observation.v1, game-result.v1
```

The frontend must never be the source of truth for legality. It can render
candidate moves and previews, but all move acceptance should come from the core
rules API.

## First Implementation Slice

1. Add a small game-session service with endpoints or callable functions for:
   - new game,
   - legal moves,
   - apply move,
   - board state/QFEN,
   - choose CPU move,
   - export completed game.
2. Build human-vs-human on top of that session API.
3. Add human-vs-CPU by selecting one existing engine adapter.
4. Add CPU-vs-CPU autoplay controls: start, pause, step, speed, seed, engine pair.
5. Add optional export of completed CPU-vs-CPU games as `game-result.v1` and,
   when root search data is available, `observation.v1` or `selfplay.v1`.

## Model Integration

A trained model should appear to the frontend only as another engine adapter.
The UI should not know PyTorch, ONNX, or checkpoint internals. The adapter should
load a `model-checkpoint.v1` manifest, validate compatible tensor/action
contracts, and expose the same `choose_move` interface as MCTS/minimax/beam.

## Data Flywheel

Once the UI exists, useful human and CPU games can feed the same model-building
pipeline:

```text
human/cpu games -> game-result.v1
cpu search positions -> observation.v1
book-guided autoplay -> selfplay.v1
contract rows -> quantik-models-materialize -> training views
```

Human games are initially evaluation/calibration evidence. They become supervised
training data only if paired with position traces or later search labels.
