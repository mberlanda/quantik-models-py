# Opening coverage: the train/test partition, decided before anything is labelled

> **Status: proposal. A human decides this before W2 is dispatched.** Nothing
> is labelled, trained or changed by this document. Every measurement below was
> read from files on the author's machine on 2026-09-20 (`runs/` is gitignored,
> so none of it can be re-checked from a fresh clone); the commands that
> reproduce each number are in the last section.

QW-021 W1. The initiative's premise is that training on plies 4-6 consumes the
held-out probe and the 99.63% figure with it. Reading the code and the corpora
shows that premise is **partly wrong, in both directions**, and that changes
which decisions matter. Read "What the evidence changes" before the decisions.

## Decisions needed

Each is argued in its own section below. **RECOMMENDED** marks the option this
paper proposes; the rejected options are written down beside it.

1. **Scope of the solve.** Frontier 6 as the packet says, frontier 5, or plies
   0-2 only?
   **RECOMMENDED: frontier 5** (exact values plies 0-5, exact policy plies 0-4,
   about 70 minutes to finish), with frontier 6 conditional on the arena result.
2. **How the probe is protected.** Exclude by canonical key, redraw a fresh
   probe, or hold out a structural family?
   **RECOMMENDED: exclude the existing 7,800-position `probe-large.jsonl` by
   canonical key at every ply**, applied to the merged result. It is the
   existing default; this decision makes it the written contract.
3. **What happens to the old 640-position probe and the 99.63%.**
   **RECOMMENDED: retire it as a held-out set today, keep the number as a dated,
   scoped historical figure.** It already stopped being held out when v2 was
   built; W-anything does not do that.
4. **How the new rows enter training.** The default ply-balanced sampler would
   give the 55 new positions at plies 0-2 about **21% of all training samples**.
   **RECOMMENDED: keep the default for the primary arm** (it is what every
   baseline was trained with), pre-register the expected deep-band cost, and
   name the ablation that runs if it is exceeded.
5. **The judging comparison, fixed now.** Which arena, which baselines, which
   start depths, what counts as a pass?
   **RECOMMENDED: matched-budget head-to-head against `patience-cpool-v3`,
   policy arena at start plies 2, 3 and 6, plus `minimax-d2` and `uniform-mcts`
   controls; a stated decision rule and a stated consequence of a null.**
6. **The opening book stays preferred, and who builds that.** Acceptance
   criterion 7 has no work item that can satisfy it.
   **RECOMMENDED: an exact-lookup layer in the play service for the plies the
   solve makes exact, network after; add a work item, because none of W2-W7's
   `allowed_paths` covers the service.**

Two scope problems with the packet itself, for the coordinator, are in
"Flags on the packet" at the end.

## What the evidence changes

**F1. The packet's coverage premise is stale.** The initiative says the flagship
trained on "zero positions below ply 6". That is the v1 corpus
(`runs/coverage.md`, dated 2026-08-27). The nested v2 and v3 corpora
(`docs/corpora.md`; ADR 0014) already reach plies 3-6. Measured on
`exact-sampled-v3.npz` against the enumerations in `runs/canonical/`:

- ply 0, 1, 2: **0 of 1, 0 of 3, 0 of 51** canonical positions. These are the
  only plies no corpus reaches.
- ply 3: 726 of 726, all policy-labelled. Complete.
- ply 4: 9,746 of 10,946 (89%); **the missing 1,200 are exactly the probe**.
  Only 983 of those rows carry a policy label; the rest are value-only.
- ply 5: 29,887 of 105,632 (28%); 4,899 policy-labelled.
- ply 6: 170,094 of 901,916 (19%); 55,068 policy-labelled.

So "extend to new plies, not densify covered ones" (acceptance criterion 2, ADR
0014) has a sharp reading: the genuinely new coverage is **55 positions at
plies 0-2**. Everything the packet adds at plies 5-6 is densification of a band
that v3 already touches, and v3 is the counterexample ADR 0014 cites.

**F2. Probe consumption is not forced.** `probe-large.jsonl` holds 1,200
positions at each of plies 4, 5 and 6 (`scripts/build_probe.py:33`), which is
11.0%, 1.1% and 0.13% of those levels. Excluding it by canonical key leaves
9,746, 104,432 and 900,716 training positions at plies 4-6. The merge path
already does exactly this: `merge()` drops held-out keys from the merged result
(`src/quantik_models/data/merge_corpus.py:56`, dropped at `:84`), and
`main()` defaults the probe set to `probe-large.jsonl` (`:119-121`). The
premise that plies 4-6 *are* the probe holds only for a run that bypasses
that exclusion.

**F3. The 99.63% is not the probe the lineup is judged on.** It comes from the
original 640-position probe (`runs/oracle/probe.jsonl`: 40, 40, 80 positions at
plies 4, 5, 6 and 80 at each of plies 7-12), for `qnet@200ms` against the v1
corpus (`scripts/build_report.py:550`). Every later comparison, ADR 0014
included, uses the 7,800-position `probe-large.jsonl`, built to be disjoint from
both the corpus and the 640 (`scripts/build_probe.py:36-50`, `:87`; verified
overlap between the two probes: 0).

The 640 is **already consumed**. Against `exact-sampled-v3.npz`, 86 of 640 are
now training rows: **40 of 40 at ply 4**, 17 of 40 at ply 5, 11 of 80 at ply 6.
(v1 held 16, the ones the audit article names.) The 99.63% stopped being a clean
generalization test for any v2 or v3 checkpoint before this initiative existed.

**F4. Frontier 6 changes the mix, not just the size.** `solve_opening.py`
gives the frontier ply value-only rows (`scripts/solve_opening.py:195-199`) and
exact policy one ply shallower. At frontier 6 the merged corpus gains about
730,600 ply-6 rows and 74,500 ply-5 rows, taking ply 6 from 4.9% to roughly 21%
of rows, and the training run's row count from 3.52M to about 4.3M. Solve costs
are nearly flat per level (status.md, measured 2026-08-30: ply 4 about 2.2 h,
ply 5 2.9 h, ply 6 4.5 h full level), and the resumable state on disk is
`opening5` 64,000 of 105,632 solved and `opening` 10,000 of 901,916 solved
(line counts of the two `frontier.jsonl` files).

**F5. The arena cannot see plies 0-2 as configured.** The lineup script starts
at plies 3, 6 and 9 (`scripts/evaluate_lineup.sh:84`, `:91`). Starting at ply 0
does not help: a network agent is deterministic, and the code's own measurement
is that at `--start-plies 0` a 300-game pairing holds **exactly one** distinct
game (`src/quantik_models/arena/autoplay.py:192-206`). ADR 0014 already records
that no arena on disk starts before ply 3.

**F6. The play service treats plies 0-3 as unknown territory.** It samples the
network at temperature 1.0 for the first four plies because "the policy head
was never trained on the first few plies and has no opinion there"
(`src/quantik_models/play/opponents.py:33-51`). No code in this repository
consults an exact book. `docs/corpora.md:160-167` already says to prefer the
book for opening play; nothing implements it here.

## D1. Scope of the solve

**Options**

- **A. Frontier 6 (the packet as written).** Exact values plies 0-6, exact policy
  plies 0-5 (`scripts/solve_opening.py:8-13`). About 4.5 h, 891,916 positions to
  go. Gives the most labels and is the only option that makes ply 5 policy exact.
  Cost: 730k new ply-6 rows of a distribution v3 already samples; the same shape
  as the v2 to v3 step that bought "a measured zero".
- **B. Frontier 5.** Exact values plies 0-5, exact policy plies 0-4. Finish
  `opening5`: 41,632 positions, about 70 minutes. Adds 74.5k ply-5 value rows and
  exact policy at ply 4 (983 of 9,746 rows have one today) and ply 3 and 0-2.
  Leaves plies 6 and deeper untouched, so nothing about the ply-6 band or the
  deep-band comparison moves.
- **C. Plies 0-2 only, no solve.** v3 holds all 726 ply-3 positions with exact
  policy, so plies 0-2 back-induct from it in seconds (QW-027 status: 55 of 55
  positions induced; the independent oracle confirmation is still open). Zero
  compute, zero densification, zero probe contact. Cost: exact ply-4 policy, where
  the contest lives, is not added at all, and the 55 positions are the whole
  treatment.

**Tradeoffs.** C is the purest reading of ADR 0014 and the cheapest, but it tests
only whether 55 rows can move ply 0-2 play, and it ships nothing for ply 4. A is
the largest treatment and the least attributable: if it wins, the win cannot be
assigned to "new plies" versus "more ply 5-6". B changes the two things the
initiative cares about (plies 0-2 exist; ply-4 policy becomes exact) and keeps the
densification to one ply, at a cost of about 70 minutes.

**Rejected:** running frontier 6 first "because it is only 4.5 h". The hours are
not the cost; the cost is a result nobody can attribute, which is what the
ADR was written to prevent. Also rejected: solving ply 6 and skipping ply 5
(policy at ply 4 needs every ply-5 child solved).

**RECOMMENDED: B**, run as the treatment, and escalate to A only if B's arena
result (D5) is a win that then wants ply-5 policy. C's rows are a strict subset
of B's, so nothing is thrown away by choosing B; it can be run first as W3 as the
packet already sequences it. Evidence: F1, F4; `scripts/solve_opening.py:70-121`
(resumable solve), `:125-160` (back-induction).

## D2. How the probe is protected

**Options**

- **A. Exclude `probe-large.jsonl` by canonical key, at every ply, on the merged
  result.** Training set at ply p is `keys(level p) minus keys(probe-large)`.
  Already the default (`merge_corpus.py:119-121`). Keeps the 7,800 positions and
  every published comparison (ADR 0014's v1/v2/v3/patience rows) on common
  ground. Costs: 1,200 positions of ply 4 (11%) stay out of training forever, so
  any statement of "complete coverage of ply 4" must say "excluding the probe".
  And the probe is a uniform random slice of each level, so it measures
  interpolation inside a band the network is trained on almost fully, not
  generalization to a band it has never seen. The claim it supports is narrower
  than the old one and must be worded so.
- **B. Retire `probe-large`, draw a fresh probe from the solved levels.** With
  the levels fully solved, a new random hold-out is free. Cost: every existing
  number stops being comparable, the four earlier checkpoints would need
  re-scoring on it, and a fresh sample is again a random slice, so it buys no
  extra rigor over A.
- **C. Structural hold-out**, for example all descendants of one ply-2 or ply-3
  class. The only design that tests extrapolation. Rejected here: a position has
  many ancestors, so excluding descendants removes far more than a slice (all of
  plies 0-3 are ancestors of some probe position, so *key* exclusion cannot use
  ancestry either), it needs a new probe and new solves, and it is incomparable
  with everything on disk.

**Residual leakage that A accepts, stated so it is not discovered later.** The
ply-3 parents of probe positions are trained, and their exact policy labels encode
that a move into the probe position wins or loses. That is the same relationship
v2 and v3 already carry (v3 shares zero keys with the probe, verified by
`docs/corpus-v3.md`, yet trains every ply-3 parent), so it does not change the
comparison, but it means the probe measures "unseen position", not "unseen
subtree".

**RECOMMENDED: A.** If the reviewer wants an extrapolation claim, C is the
follow-up, as its own initiative. Evidence: F2; `scripts/build_probe.py:36-50`;
`merge_corpus.py:8-14` (why exclusion is on the merged result).

## D3. The old 640 probe and the 99.63%

**Options**

- **A. Retire it as a held-out set; keep the figure as a dated, scoped
  historical statement.** "99.63% on a 640-position probe, `qnet@200ms`, v1
  corpus, 2026-08-27." Do not re-measure, do not exclude it from the new corpus.
- **B. Reconstruct it.** Retrain an arm with the 640 excluded and re-measure.
  Rejected on evidence, not taste: 86 of the 640 are already in v3 (F3), so an
  exclusion changes the corpus and confounds the comparison that D5 needs clean,
  for a number nobody ranks on.
- **C. Retain it as a live probe** by excluding its 160 ply 4-6 positions too.
  Rejected for the same reason, and because 68 of those 160 are already trained
  on; excluding them removes rows v3 has.

**RECOMMENDED: A.** The decision the packet asked for ("retired, retained,
reconstructed; ambiguity is not acceptable") is: **retired as a probe, retained as
history.** The sentence that replaces the bare number wherever it appears is:
"measured on the 640-position original probe against the v1 corpus; that probe
has since been partly trained on (86 of 640 positions) and is no longer held
out". Where the figure lives: `articles/part-vii-the-audit.md:124` and
`articles/preview.html:291` (articles repo, W7), and
`scripts/build_report.py:550` (this repo; see flags).

## D4. How the new rows enter training

`TrainConfig.balance_plies` defaults to `True` (`src/quantik_models/train/supervised.py:84`),
and `ply_sampling_weights` makes every ply equally likely per draw
(`src/quantik_models/data/exact_corpus.py:148-153`, used at
`supervised.py:219`). v3 has plies 3-13, eleven values. Adding plies 0, 1 and 2
makes fourteen, so plies 0-2 receive 3 of 14 = **21.4% of all samples from 55
rows**, and every other ply drops from 9.1% to 7.1% of the gradient.

That is a large, silent effect: memorising 55 exactly-solved positions is
arguably what one wants, but 22% fewer samples on plies 3-13 is the same
"shallow data costs the deep band" trade ADR 0014 measured (v3's deep band moved
-0.0047, p = 0.033).

**Options**

- **A. Keep `balance_plies=True`.** The recipe every baseline used, so the
  comparison stays matched. The treatment then *includes* the sampler shift,
  which is honest but bundles two effects.
- **B. Cap the mass of the new plies** (for example plies 0-2 together at no more
  than 2%). Isolates coverage from re-weighting, but needs a training-code change
  none of W2-W7 is permitted to make, and it leaves the arm unmatched to every
  baseline.
- **C. `balance_plies=False`.** Removes the shift, but the natural distribution
  gives 55 rows a vanishing share and would very likely produce a null at plies
  0-2 by construction, a false negative on the whole initiative.

**RECOMMENDED: A for the primary arm**, with a pre-registered tripwire: if the
deep band (plies 7-12 on `probe-large`) falls by more than 0.5 points, that is
reported as a cost of the treatment, and the B-style ablation is the named
follow-up rather than a reason to reinterpret the primary result afterwards.
Whether `balance_plies` is a CLI flag was not checked; W5 must confirm before it
relies on either arm.

## D5. The comparison that judges the result

Held-out policy accuracy has failed to predict play strength five times (ADR
0014 counts the fifth). It is therefore reported, never the criterion.

**Fixed now:**

- **Candidate.** `cpool`, same architecture, learning rate (6e-4), seed
  (20260828), epoch cap and patience as `patience-cpool-v3`, on the merged corpus
  (v3 plus the D1 rows, minus the D2 exclusion). This is the matched-budget
  discipline of ADR 0014: corpus is the only variable.
- **Baselines.** `patience-cpool-v3` (matched cap; primary), `v3-cpool` at 16
  epochs, `minimax-d2`, and the `uniform-mcts` control the lineup script already
  includes (`scripts/evaluate_lineup.sh:72`).
- **Start depths.** Policy arena at start plies **2, 3 and 6**, 300 games per
  ordered pairing; MCTS-128 at 2, 3 and 6 as the second condition. Ply 2 is new
  and is the only depth that can see the treatment; ply 3 and 6 are the depths
  ADR 0014's comparisons already used, so the result stays comparable. Ply 0 is
  excluded by F5. Use a start-position sample, not a fixed suite, because
  `--start-plies 2` needs no code change (`autoplay.py:157-176`) and W5 is only
  permitted to edit docs and `scripts/evaluate_lineup.sh`. An exhaustive suite of
  the 51 canonical ply-2 positions is cleaner and is the option if the
  coordinator widens W5 to the arena.
- **Arena seed.** Not a training seed, as `evaluate_lineup.sh:33-38` requires;
  report the distinct-game count next to every rate (`autoplay.py:192`).

**Decision rule (a proposal for the reviewer to change now, not after).**

- **Pass:** the head-to-head win rate against `patience-cpool-v3` has a 95%
  interval above 50% at start ply 2, and is not significantly below 50% at start
  ply 6, under both policy and MCTS-128.
- **Null:** intervals straddle 50% at ply 2. Then the initiative reports a
  negative result and does **not** proceed to frontier 6. This is stated in
  advance because the v2 to v3 result was rationalised after the fact once.
- **Regress:** significantly below 50% at ply 6 or a deep-band probe drop over 0.5
  points (D4 tripwire). Report as the cost of the treatment.

**Options considered.** (a) Probe accuracy as primary: rejected, five failures.
(b) A fit check on all 781 canonical positions at plies 0-3 against the exact
optimal sets: kept as a **secondary report only**, because these positions are
training data by construction, so it measures fit, not generalization. (c) The
arena as above: **RECOMMENDED**.

Cost note for the reviewer: 300 games times the pairings above at three depths is
the same shape as `evaluate_lineup.sh` today, so the arena cost is known from
`runs/eval/`; the training run is the expensive line.

## D6. The opening book stays preferred, and who builds it

**Options**

- **A. Status quo:** network at temperature 1.0 for the first four plies
  (`opponents.py:33-51`). After this expansion that comment becomes false (the
  network has an opinion at plies 0-4) but the behaviour would silently persist.
- **B. Exact lookup where exact, network after.** `solve_opening.py` emits an
  exact optimal-move mask for every canonical position at plies 0-4 (frontier 5)
  or 0-5 (frontier 6), keyed by canonical key
  (`scripts/solve_opening.py:195-231`). Serve those answers from a lookup; sample
  uniformly over the exact optimal set to keep variety (the reason the
  temperature exists); the network answers only beyond. Depth of the book follows
  D1.
- **C. Trust the stronger network everywhere.** Rejected by QW-021 decision 4:
  the book is exact, the network at best approximates it.

**RECOMMENDED: B.** The Rust opening book already exists in `quantik-core-rust`
(workspace repository map); whether the service should call that or read the
`opening-exact.npz` this solve produces is a design question for whoever owns the
work item, and this paper does not decide it.

**This criterion currently has no owner.** No work item in the manifest can edit
`src/quantik_models/play/`. See the flags.

## Reproducing the partition

The partition is a set difference on canonical keys, no random numbers. From the
`quantik-models-py` root with `runs/` present:

```python
import numpy as np
from pathlib import Path
from quantik_models.env import fastboard as fb
from quantik_models.data.merge_corpus import probe_keys

held = probe_keys([Path("runs/oracle/probe-large.jsonl")])   # 7,800 keys
for ply in range(1, 7):                                      # level0N.npy is ply N
    keys = fb.canonical_keys(np.load(f"runs/canonical/level{ply:02d}.npy"))
    n_held = int(np.isin(keys, np.fromiter(held, dtype=keys.dtype)).sum())
    print(ply, len(keys), n_held, len(keys) - n_held)
```

Output on 2026-09-20, which W2 must reproduce before it starts:

- ply 1: 3 positions, 0 held out, 3 train
- ply 2: 51, 0, 51
- ply 3: 726, 0, 726
- ply 4: 10,946, 1,200, **9,746**
- ply 5: 105,632, 1,200, **104,432**
- ply 6: 901,916, 1,200, **900,716**

Ply 0 is the empty board and has no level file (`scripts/solve_opening.py:44-45`, the
enumeration starts at ply 1). Other measurements: corpus coverage in F1 comes from
`ExactCorpus.load(...).canonical_keys()` intersected with the same levels; the old
probe overlap in F3 from the same call on `probe.jsonl`.

## Not verified here

- The QW-027 independent oracle confirmation of the induced plies 0-2 values is
  still open; D1's option C and the ply 0-2 rows under B inherit that caveat.
- The per-level solve timings are status.md's, not re-measured; this work did no
  solve.
- Whether `balance_plies` is exposed on the command line (D4).
- Arena costs for the D5 conditions.

## Flags on the packet, for the coordinator

- **W6's `allowed_paths` do not contain the figure.** `docs/models.md`,
  `docs/labeling-strategy.md` and `README.md` do not quote 99.63%; the only
  occurrence in this repository is `scripts/build_report.py:550`, which W6 may not
  edit. Either widen W6 or record that this repository has nothing to rescope.
- **Acceptance criterion 7 has no work item.** None of W2-W7 may edit
  `src/quantik_models/play/`. Add a W8 or drop the criterion.
- **This document adds one line to `docs/README.md`**, outside W1's single
  allowed path, because `tests/test_docs_crossrefs.py::test_the_index_links_every_document`
  fails on any `docs/*.md` the index does not link. Without it `main` goes red.
