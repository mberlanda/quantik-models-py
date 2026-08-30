# Autoplay: what it is for, and what the games said

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


> **Restated 2026-08-30 at swept learning rates.** `cpool` was trained at
> 2e-3 — a rate chosen for the ResNet — and prefers 6e-4. Retraining it
> reversed the arena conclusions below, not just their decimals. The
> superseded reading is kept at the end of this document.

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

```bash
# 3. fold them into the corpus a trainer reads
python -m quantik_models.data.merge_corpus \
  --corpus runs/oracle/corpus/exact-sampled.npz \
  --solved runs/autoplay/lineup-p3/solved.jsonl \
  --out runs/oracle/corpus/exact-sampled-v2.npz
```

The merge enforces the two invariants everything downstream assumes. **The
probe stays held out** — exclusion is applied to the merged result, not
only to the new rows, because solving a position also labels its children
and a probe position can arrive as somebody's child without ever being
sampled. That is exactly how sixteen probe positions reached the first
corpus. And **one row per canonical position**, with policy-labelled rows
winning the tie-break over value-only ones, so a position solved in one
file and derived as a child in another is not counted twice.

### Human games feed the same queue

Autoplay explores where the *engines* go. The play service
(`play/server.py`) records games humans actually play, and every recorded
game already has its positions in `game_positions` — but nothing consumed
them until `play/export.py`, so positions people actually reach never
joined the corpus. It closes that loop by producing the same artifact:

```bash
python -m quantik_models.play.export \
  --db ~/.local/share/quantik/games.db \
  --corpus runs/oracle/corpus/exact-sampled-v3.npz \
  --out runs/play/packed \
  --max-ply 6
```

Same rule as above, stricter: **human game outcomes are never labels, only
positions travel.** A human game's `winner` says which of two fallible
players won, not the value of a position, so the exporter never reads
`games.winner` — it reads `distinct_positions` and nothing else. The one
trap is that `game_positions.canonical_key` is stored as a decimal string
(`play/record.py:_canonical_key`) while `ExactCorpus` gives a `uint64`
array; comparing them without converting finds no overlap and silently
queues positions the corpus already has. `play/export.py`'s module
docstring is where that conversion happens, once.

Run against the live store (20 games, 161 positions, 101 distinct keys at
the time of writing): 66 positions at ply ≤ 6, 40 already in
`exact-sampled-v3.npz`, 26 written to the queue. Fed straight into
`exact_oracle` and `merge_corpus.py` with no format change — solving a
ply-≤6 position is far more expensive than the deeper ones autoplay
usually queues (see "Plies 0–3 are unevaluated" in the workspace's
WORKSTREAMS.md), so this queue is small and the run that actually clears
it is separate, future work.

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

### The seed was doing nothing at all (2026-08-29)

The paragraph above understated the problem, and the understatement lasted
until somebody watched two models play in the browser and noticed every
game opening the same way.

Every agent takes `select(board, seed)`, and `arena.match`, `arena.autoplay`
and the play service all thread a varying seed through it. For the
network-backed agents that seed decided **nothing**:

- `PolicyAgent` defaults to `temperature=0.0`, which is an argmax over the
  priors.
- `NetMCTSAgent` returned `visits.argmax()`, and the only consumer of its
  RNG inside `BatchedMCTS` is the root Dirichlet noise — which every arena
  spec disables with `dirichlet_weight: 0.0`.

So `uniform-mcts128` returns action 0 from an empty board for each of six
different seeds, and `--start-plies 0 --games N` produced N byte-identical
games per pairing. Not "roughly two distinct games": exactly one.

**What this cost the published numbers.** Nothing was invalidated, because
every lineup run used `--start-plies 3`, `6` or `9`, and the random starts
carried the variation. What it cost is *effective sample size*, which no
output reported. Counting distinct action sequences per ordered pairing in
`runs/eval/lineup-2026-08-29`:

- at `--start-plies 3`, 298-300 of 300 games were distinct;
- at `--start-plies 6`, **263-272 of 300**.

`run` samples start positions with replacement, so a pairing gets fewer
distinct openings than it asked for, and a repeated opening between two
deterministic agents is a repeated *game*. The ply-6 intervals in
`arena.pack` divide by 300 and should divide by something nearer 265. The
effect is small — roughly a 6% understatement of the interval width — but
it was invisible, which is the part worth fixing.

`autoplay` now prints the distinct-game count beside the leaderboard,
writes it into `games.json` per pairing, and warns when the worst pairing
falls below half. It reports the shortfall rather than silently correcting
the intervals; what the effective sample size *is* under repeated openings
is a separate argument.

### The fix, and the three alternatives

`NetMCTSAgent` and `PolicyAgent` now take `temperature` and
`temperature_plies`. At a non-zero temperature the move is **sampled from
the MCTS visit counts** (or the priors, for the policy agent) instead of
taken as an argmax, and `temperature_plies` bounds that to the opening. The
default is `0.0` — unchanged, deterministic, which is what every published
margin was measured at.

Sampling the **visit counts** rather than the priors, because the visits
are what the search converged on: a temperature there trades strength for
variety along the search's own ranking, instead of discarding the search.

Three alternatives, and why not:

- **Random tie-breaking on the argmax.** Nearly free and strictly
  harmless, and it buys nothing — measured, `uniform-mcts128` from an
  empty board has exactly one action at its maximum, and `cpool@128` has
  102 visits on its best action against 1-2 on nineteen others. There are
  no ties to break.
- **Root Dirichlet noise**, which already exists and is already disabled.
  It is AlphaZero's own diversity mechanism, and it would fix the case
  temperature does not (see below). It was not chosen as the primary
  mechanism because it perturbs the priors the whole tree is built on, so
  its cost is spread through every simulation rather than landing on one
  choice — harder to reason about, and harder to bound to the opening.
- **Randomised starts only**, the status quo. It cannot reach plies 0-2 at
  all, and the pre-plies are uniformly random rather than plausible, so the
  positions it reaches are drawn from a wider distribution than real play —
  a caveat this document already carries at the bottom.

### A thing found on the way: the uniform control barely searches

`uniform-mcts128` is the control that separates "the network is good" from
"the search is doing the work". With uniform priors *and* a flat zero
value, PUCT is degenerate: the first child visited sits at `Q = 0` while
every unvisited sibling sits at `-fpu_reduction`, and at `c_puct = 1.5`
with a prior of 1/64 the exploration bonus never closes that gap.

Measured at 64 simulations with `fpu_reduction=0.2`: **all 64 visits land
on one action**. At 128 simulations, 113 of 128 on one action across 12
actions total. A trained checkpoint spreads much further — `cpool` at 128
simulations puts 102 on its best and touches 20 actions.

The control is therefore weaker than "the same search without a network"
suggests; it is closer to "the same search, collapsed onto its first
descent". That does not invalidate its use as a floor, but it does mean the
gap between a network and `uniform-mcts128` overstates the network's
contribution relative to a search that actually explores. Setting
`fpu_reduction: 0.0` in its spec makes it fan out to 25 actions. **Not
changed here** — doing so would change what the published control means,
and restating that comparison is its own piece of work.

## What the first full cycle produced

Run on 2026-08-29: 2,400 games from ply-3 starts, 5,226 novel shallow
positions, solved exactly in **6h50m** on twelve threads, then merged.

The solve turned 5,226 solved parents into **118,053 labelled rows** —
roughly twenty-two free child labels each, which is what makes the cost
worth paying.

| ply | corpus before | corpus after | |
|---|---|---|---|
| 3 | 0 | **664** | new |
| 4 | 0 | **9,664** | new |
| 5 | 0 | **22,655** | new |
| 6 | 40,000 | **86,631** | 2.2x |

3,087,356 -> 3,196,958 rows; 250,000 -> 255,058 policy-labelled. This is
the direct answer to `shift-evaluation.md`: plies 0-5 held *zero* training
positions, which is why every architecture was weakest there and why the
arena ranking flips with start depth. There is now exact, solver-labelled
data in that region, reached by the engines' own play rather than by
sampling the canonical space.

### The held-out guard earned its place immediately

The merge **dropped 1,554 probe positions**. They arrived as *children* of
solved positions — never sampled, never chosen, labelled for free as a side
effect of solving their parents. That is exactly the mechanism
`data/merge_corpus.py` documents, and the reason exclusion is applied to
the merged result rather than only to the new rows.

Without it the probe would have been contaminated and every number in
`shift-evaluation.md` would have quietly become part recall. Verified
after the fact: the v2 corpus shares **zero** canonical keys with the
probe, and all 3,196,958 rows are distinct canonical positions.

Note this is a *different* experiment from the architecture comparison.
Retraining on v2 measures a better corpus, not a better architecture, and
conflating the two would make both unreadable.

## The arena result

These are all *relative* results — network against network. For the same
models against a fixed classical opponent, see `oracle-benchmark.md`: the
ranking is the same and the margins are wider.


3,600 games per start depth, every ordered pairing, seed 20260830 — chosen
to differ from every training seed so that seed-linked bias would show
rather than hide.

| start ply | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| 3 | **`cpool` 57.2%** | `attn` 54.2% | `resnet` 47.8% | `mlp` 40.8% |
| 6 | **`attn` 54.3%** | `cpool` 51.3% | `resnet` 48.8% | `mlp` 45.5% |
| 9 | **`cpool` 52.2%** | `attn` 50.2% | `resnet` 49.5% | `mlp` 48.1% |

Significant head-to-heads (Wilson 95%): `cpool` beats the ResNet 58.5% at
ply 3 and 54.7% at ply 9; `attn` beats the ResNet 54.8% at ply 3 and 54.3%
at ply 6; both beat the MLP everywhere.

**On raw policy the ResNet is now third at every depth.** `cpool` and
`attn` trade the lead between them, and the ResNet is not in the argument.
The scope of that sentence matters: under search it comes *second* at both
depths, for the reason given below.

## Under search, the differences are real

6,000 games per depth at 128 simulations, with the uniform-prior control.

| start ply | order | |
|---|---|---|
| 3 | `cpool` 67.8% · `resnet` 63.9% · `mlp` 59.0% · `attn` 58.6% | `uniform` 0.7% |
| 6 | `attn` 58.3% · `resnet` 57.8% · `cpool` 56.8% · `mlp` 55.5% | `uniform` 21.5% |

At ply 3, `cpool` beats the ResNet 54.2% and the MLP 60.5%, both
significant. All four networks beat the uniform control by 99%+.

**And a genuinely new finding: search *hurts* `attn` relative to the
others.** It is second on raw policy at ply 3 and last among the networks
under search, losing to both `cpool` (42.7%) and the ResNet (42.8%). Search
leans on the leaf value, and `attn`'s value head is measurably weaker —
0.0378 MAE against `cpool`'s 0.0315 in training, 0.0881 against 0.0777 on
the shift probe. Good priors, weaker values, and search finds the
difference.

## Restated 2026-08-30 under patience-based budgets

Same four checkpoints, retrained with `--patience 5 --epochs 60` in place
of the fixed 16-epoch budget above. `scripts/evaluate_lineup.sh`, output
`runs/eval/patience-2026-08-30/`.

> **Seed caveat.** This run took the script's default seed, `20260829`,
> because nothing overrode it — the same value already spent on
> `runs/eval/epoch-test/`, a different comparison. The workspace task that
> commissioned this run (QW-012) explicitly called for a fresh arena seed
> for exactly the reason the script's own comment gives: reusing one makes
> seed-linked bias invisible rather than absent. Caught after the ~3.6h run
> finished, not before. The ranking below is not being withdrawn on the
> strength of this alone, but it has not been confirmed on an independent
> seed either — treat a re-run on a fresh seed as unfinished business, not
> optional polish. Tracked as `QW-026` in `quantik-workspace`.

On raw policy (`net-policy`, deterministic — the ply-9 row is directional
only, since its worst pairing replays just 135 of 300 distinct games):

| start ply | 1st | 2nd | 3rd | 4th | distinct games (worst pairing) |
|---|---|---|---|---|---|
| 3 | `cpool` 55.0% | `attn` 52.2% | `mlp`/`resnet` 46.4% (tied) | | 297/300 |
| 6 | `attn` 52.6% | `cpool` 51.9% | `resnet` 50.8% | `mlp` 44.8% | 244/300 |
| 9 | `cpool` 51.1% | `attn` 50.3% | `mlp` 49.6% | `resnet` 49.0% | 135/300 |

Under 128-simulation MCTS, where distinct-game coverage is far better:

| start ply | order | uniform control | distinct games (worst pairing) |
|---|---|---|---|
| 3 | `cpool` 66.4% · `resnet` 61.8% · `mlp` 61.5% · `attn` 59.9% | 0.5% | 292/300 |
| 6 | `attn` 58.4% · `cpool` 57.6% · `resnet` 57.0% · `mlp` 56.1% | 21.0% | 230/300 |

The cpool/attn "tie" ADR 0001 reported on fixed-budget top-1 does not
survive this arena: `cpool` wins ply-3 MCTS by 6.5 points, the widest gap
in either table, and only loses ply-6 MCTS to `attn` by 0.8 points —
inside the noise a single seed and imperfect game diversity can produce,
not a second regime the way the ResNet-vs-`cpool` split above was.
`resnet`/`mlp` stay mid-pack under MCTS despite gaining the most top-1 from
the longer training budget, which is this document's point restated: raw
policy accuracy and play strength are separate measurements.

## What the earlier version of this document claimed, and why it was wrong

Before `cpool` was retrained, this file reported that the ResNet led from
ply-3 starts at 53.7%, that "each network's advantage is real and lives at
a specific depth", that the ResNet owned the opening and `cpool` the
midgame, and — separately — that 128 simulations of search **flattened
every difference**, with no significant head-to-head at either depth.

None of those survive. The ResNet's opening lead was `cpool` being
undertrained; at the correct rate `cpool` beats it at ply 3 by a
significant margin. And search does not flatten the field: several pairings
are now significant that previously were not.

The one claim that does survive, in weakened form, is depth dependence —
`attn` leads at ply 6 and `cpool` at plies 3 and 9. But it is a
`cpool`-versus-`attn` effect, not the ResNet-versus-`cpool` story that was
written down.

## What this still does not establish

- **No baseline.** None of these played the minimax or MCTS engines, so
  every rate here is relative to the other two networks.
- **Random starts, not played ones.** `--start-plies N` plays N *random*
  legal moves. Positions an engine would actually reach at ply 6 are a
  different, narrower distribution, and the numbers could differ there.
