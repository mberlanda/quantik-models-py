# Autoplay: what it is for, and what the games said

## It generates positions, not labels

Autoplay is usually described as a way to make training data out of game
results: play, take the outcome, train on it. That is not what it does
here, and the distinction is the whole design.

This project already has better labels than any game can produce. The
corpus carries the true game-theoretic outcome and the full
outcome-optimal action set, from an exact solver. A game result is a much
weaker signal: the value is contaminated by both players' mistakes, and the
"policy target" is one move that may simply be wrong. The AlphaZero run in
this repo already paid that price — its value head learned almost nothing,
because the target blended an 8-ply game result with its own undertrained
estimate.

What autoplay uniquely provides is **reach**. The corpus spans plies 6-13.
Games spend their opening moves at plies 0-6, where there is not one
training position, and where `shift-evaluation.md` shows every architecture
is at its weakest. Those positions are also *reachable in real play*, which
uniform sampling of the canonical space does not guarantee — a position no
engine would ever walk into is not worth a solver call.

So the pipeline is:

```bash
# 1. play, and keep the positions
python -m quantik_models.arena.autoplay \
  --agents runs/arena/lineup-agents.json \
  --games 400 --start-plies 3 --out runs/autoplay/lineup-p3

# 2. label them exactly
../quantik-core-rust/target/release/examples/exact_oracle \
  < runs/autoplay/lineup-p3/to-solve.qfen \
  > runs/autoplay/lineup-p3/solved.jsonl
```

Positions deduplicate on the **canonical key**, and anything the corpus
already holds is dropped, because solving is the expensive step: about 5.5
minutes per hundred positions at these depths on twelve threads.

### Deterministic agents need randomised starts

`net-policy` takes the argmax with no temperature, so two games between the
same pair from the same position are the *same game*. A first run of 30
games from the empty board produced 45 distinct positions — roughly two
distinct games per pairing.

`--start-plies N` plays N random legal moves before the engines take over.
At `--start-plies 3`, 2,400 games produced **10,587 distinct positions**,
of which 5,226 were both shallow and novel. The cost is that plies 0-2 are
never visited; a run from the empty board is still the right way to see
what the engines actually open with.

## The arena result

2,400 games, every ordered pairing, `--start-plies 3`, seed 20260829.
Ordered pairings because moving first is a real advantage in Quantik, and a
pairing played only one way round would credit that advantage to the agent
rather than to the seat.

| agent | win rate | record |
|---|---|---|
| `resnet-c128-b6` | 53.7% | 859/1600 |
| `cpool-c191-b6` | 49.9% | 798/1600 |
| `mlp-h455-b4` | 46.4% | 743/1600 |

Head to head, with Wilson 95% intervals:

| pairing | record | rate | 95% CI | |
|---|---|---|---|---|
| `resnet` vs `mlp` | 456–344 | 57.0% | [53.5%, 60.4%] | **significant** |
| `resnet` vs `cpool` | 403–397 | 50.4% | [46.9%, 53.8%] | not significant |
| `cpool` vs `mlp` | 401–399 | 50.1% | [46.7%, 53.6%] | not significant |

## Reading it against the other two measurements

This is now the third independent look at the same three networks, and
they do not agree:

| | IID top-1 | shift, deep | shift, shallow | arena |
|---|---|---|---|---|
| `resnet-c128-b6` | 0.9701 | 0.9720 | **0.9126** | **53.7%** |
| `mlp-h455-b4` | 0.9516 | 0.9578 | 0.8843 | 46.4% |
| `cpool-c191-b6` | **0.9851** | **0.9883** | 0.9092 | 49.9% |

**`cpool` leads two of the four and wins none of the games.** It is 6.6
points better than the MLP on shift-deep accuracy and has *half* its value
error, and it cannot beat it: 401–399 over 800 games is as close to a coin
flip as the measurement can resolve.

The ordering that survives into actual play is the **shallow** column, not
the headline one. That is consistent with where these games are decided:
started at ply 3 and typically over well before ply 10, they are fought
almost entirely in the region where `cpool` is weakest and `resnet`
strongest.

**Move-choice accuracy is a poor predictor of playing strength here**, and
the reason is not mysterious. Accuracy counts every position equally.
A game does not: one bad move in a sharp opening position decides it, and a
hundred correct moves in a position already won change nothing. `cpool`
spends its advantage where the game is no longer in doubt.

## What this does not establish

- **The arena does not rank `cpool` against `resnet`.** 403–397 is not a
  result. It says the two are close from ply-3 starts, nothing more.
- **One start depth.** Everything here is `--start-plies 3`. A deeper start
  would hand the game to the region `cpool` dominates, and there is a real
  chance the ordering flips. That is a cheap experiment and it has not been
  run.
- **These are `net-policy` agents** — one forward pass, argmax, no search.
  Search would let a better value head do work that pure move choice
  cannot, which is exactly `cpool`'s strength. `net-mcts` over the same
  three is the obvious follow-up.
- **No baseline.** None of these played the minimax or MCTS engines, so
  "53.7%" is a rate against the other two networks, not against the
  project's incumbent engines.
