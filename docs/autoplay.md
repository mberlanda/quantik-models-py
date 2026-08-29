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

**The ResNet is now third at every depth.** `cpool` and `attn` trade the
lead between them, and the ResNet is not in the argument.

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
