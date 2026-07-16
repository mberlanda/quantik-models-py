# Distilling Search Into a Compact Policy/Value Network for Quantik: Design, Tradeoffs, and Cross-Domain Context

*Technical report, quantik-models-py project — 2026-07. Status: design
and infrastructure report; experimental sections describe planned and
in-progress work, not completed results.*

## Abstract

We describe the design of the first neural policy/value model for
Quantik, a two-player abstract strategy game on a 4×4 board. Rather
than pure self-play reinforcement learning, we bootstrap the model by
distilling heterogeneous search artifacts — exhaustively searched
opening books, solver-verified position labels, and Monte Carlo tree
search (MCTS) visit distributions — into a small residual convolutional
network with a legality-masked 64-way policy head and a scalar value
head. The training pipeline is contract-first: every artifact that
crosses a repository boundary (observations, game results, self-play
rows, model checkpoints) is a versioned, schema-validated contract, and
the exported checkpoint is a safetensors weights file plus a
`model-checkpoint.v1` manifest that downstream engines validate before
loading. We discuss the central tradeoffs — distillation from search
versus pure self-play, network capacity versus a deliberately small
game, masking legality at the loss rather than in the architecture, and
book memorization versus learned generalization — and situate the
design in the literature line from AlphaGo Zero and AlphaZero through
KataGo's efficiency program, as well as AlphaZero-style systems applied
outside board games. We close with the planned evaluation protocol:
fixed-budget search improvement and head-to-head play against
book-only and search-only baselines on book-frontier positions.

## 1. Introduction

Quantik is a perfect-information, two-player game marketed by Gigamic:
each player owns two copies each of four shapes and places them on a
4×4 board divided into four 2×2 zones. A placement is illegal if the
*opponent* already has the same shape in the target row, column, or
zone; whoever completes any row, column, or zone containing all four
distinct shapes (regardless of ownership) wins, and a player with no
legal move loses. The game is small by the standards of the AlphaZero
literature — 64 nominal actions (`action_index = shape * 16 +
position`), boards encodable in 128 bits — yet it exhibits the
blocking/paralysis tactics that make it non-trivial: a placement both
advances one's own completion threats and burns the opponent's right
to use that shape in three regions at once.

This size profile inverts the usual economics of neural game-playing.
For Go, the network is the only tractable way to approximate a value
function; for Quantik, exhaustive search is cheap enough that the
project already maintains an IDDFS-generated opening book (SQLite,
canonical-key deduplicated) and an exact solver used to label benchmark
positions. The motivating question for a model is therefore *not*
"can we approximate the value function at all" but: **can a compact
learned evaluator generalize beyond the stored book frontier, so that
model-guided search outperforms both raw search and book lookup at a
fixed compute budget?** A book memorizes; a model interpolates. The
project's contracts documentation frames this as the 50–100 MB
policy/value model target, with the loop:

```
opening book (depth 6) -> search observations -> training view
      ^                                              |
      |                                              v
new autoplay/H2H disagreements <- model-guided search <- checkpoint
```

This report documents the design decisions behind the first training
slice (dataset loader, network, trainer, checkpoint export), the
tradeoffs they embody, and the literature and cross-domain context in
which those decisions were made.

## 2. Background: the Quantik data pipeline

The project is split across four repositories with a contract-first
discipline: `quantik-core-contracts` owns JSON-schema'd artifact
contracts and validators; `quantik-core-rust` owns high-throughput
producers (opening-book construction, position generation, MCTS /
minimax / beam observation and head-to-head runs, self-play export);
`quantik-core-py` owns validating readers; `quantik-models-py` (this
repository) owns training views, the trainer, and checkpoint export.

Three data sources feed training, in decreasing label quality:

1. **Solved references.** Benchmark positions carry exact game values
   and complete optimal-move sets from a budgeted solver, reused and
   written back through the opening book. Positions are keyed by an
   18-byte canonical key shared by up to eight board orientations
   (the D4 symmetry group of the square); to avoid serving moves in a
   wrong orientation, book reads and writes are currently restricted to
   canonical-representative boards, with symmetry-transform tracking as
   a documented follow-up.
2. **Search observations** (`observation.v1`): per-position rows from
   MCTS/minimax/beam runs with dense `policy_visits[64]` and scalar
   evaluations.
3. **Self-play** (`selfplay.v1`): MCTS root visit distributions and
   final game outcomes.

The materializer flattens these into `.npz` training views: `tensors
(n, 9, 4, 4)` float32 (8 bitboard planes + side-to-move plane),
`policy_target (n, 64)` (normalized visit counts), `value_target (n,)`
in [-1, 1], `sample_weight (n,)`, a `uint64` legal-action bitmask, and
provenance tags. Value labels follow a precedence order — exact/book >
bounded solver > strong search > generic search > self-play outcome >
heuristic — and `sample_weight` encodes that confidence, an explicit
lesson from the label-quality analyses in the AlphaZero replication
literature (ELF OpenGo documents how sensitive training is to
data-generation details [7]).

## 3. Method

### 3.1 Network

The model is a deliberately conventional AlphaZero-style residual
network [1, 2]: a 3×3 convolutional stem from 9 input planes to `C`
channels, `B` residual blocks, a policy head (1×1 convolution, flatten,
linear to 64 logits) and a value head (1×1 convolution, flatten, MLP,
`tanh`). Novel architecture is out of scope for the first slice by
design: the point is a correct, contract-validated loop, and the
AlphaZero-line consensus is that trunk capacity and data quality
dominate architectural cleverness at this scale. Polygames' fully
convolutional + global pooling variants matter when boards vary in
size [8]; Quantik's board is fixed at 4×4, so the plain
flatten-and-project heads suffice.

One family, three presets — `smoke` (16×2, <1 MB), `small` (64×4,
~1.3 MB), `target` (256×13, ~61 MB float32) — so that CI, laptop
experiments, and the contract-envelope model differ by one flag rather
than by code path. The 50–100 MB envelope is a *contract* bound (it
sizes the artifact runtimes must accept), not a claim that Quantik
needs 15M parameters; §4.2 discusses this tension honestly.

### 3.2 Loss and legality masking

Training minimizes a sample-weighted sum of (i) soft-target
cross-entropy between the visit-count distribution and the network's
log-probabilities computed over *legal actions only* — illegal logits
are forced to −inf before the log-softmax — and (ii) mean squared error
on the value target, with AdamW weight decay. Distilling full visit
distributions rather than argmax moves follows the policy-improvement
reading of MCTS: the visit distribution is the improved policy, and
matching it transfers more information per position than matching its
mode (made precise by Gumbel AlphaZero's analysis of the
policy-improvement operator [6]).

Masking at the loss/adapter level rather than baking legality into the
architecture is a deliberate contract decision: `model-checkpoint.v1`
states that runtimes must apply legal-action masks outside the model.
The trainer therefore also *reports* pre-mask illegal probability mass
as a diagnostic — how much legality the raw network absorbs is an
empirical question worth tracking, but no correctness property ever
depends on it.

### 3.3 Deterministic content-addressed sharding

Train/validation/test membership is computed per row as
`sha1(tensor_bytes || policy_bytes || source_tag) mod 100` against
80/10/10 thresholds. Membership therefore survives row reordering,
re-materialization, and corpus growth — a row can never silently
migrate from test to train between runs, which is the failure mode that
invalidates longitudinal comparisons. The cost is that exact global
split fractions are only approached in expectation; at smoke-corpus
sizes a split can be empty (the trainer falls back to validating on
train and says so in the report).

### 3.4 Checkpoint contract

Export writes flat safetensors weights plus a `model-checkpoint.v1`
manifest: architecture string, parameter count, SHA-256 weights hash,
byte size, input contracts, and pointers to the training report. The
manifest round-trips through `quantik-core-py`'s validating reader in
CI. Weights never enter the core libraries; engines that cannot satisfy
the manifest must fail fast. This is the same artifact-boundary
discipline that lets the Rust and Python stacks share SQLite books and
Parquet shards, extended to model weights.

## 4. Tradeoffs

### 4.1 Distillation-from-search versus pure self-play

AlphaGo Zero [1] and AlphaZero [2] demonstrated tabula-rasa self-play
RL; MuZero removed even the need for rules inside search [3]. But the
replication literature is equally clear about the price: thousands of
TPU/GPU-days (ELF OpenGo: 2,000 GPUs for 9 days [7]), and KataGo's
entire program is a catalog of tricks to cut that cost by ~50× [4].
For a game where exact labels are *cheap*, starting from supervised
distillation of search artifacts is the economically sane bootstrap —
this is Expert Iteration's decomposition of planning (search generates
expert targets) and generalization (the network compresses them) [5],
with the twist that our "experts" are heterogeneous: solver, book,
three search engines, and self-play, blended by the sample-weight
precedence rather than treated as one oracle. The planned second phase
(book-guided autoplay, then active learning on model/search
disagreements) re-introduces the self-play loop only where the
supervised corpus is weak — closer to ExIt's iterated schedule than to
AlphaZero's single homogeneous loop.

The risk specific to distillation is inheriting the teacher's blind
spots: single-visit observation policies (retained for smoke data,
tagged `policy:single-visit`) are near-vacuous targets, and minimax
labels at shallow depth systematically misvalue zugzwang-like paralysis
positions. Engine mixture and the precedence weighting are the
mitigations; measuring per-source validation loss is part of the
planned evaluation.

### 4.2 Capacity versus domain size

A 256×13 residual net (~15.4M parameters) against a game whose
canonical state space is small enough for exhaustive shallow books is,
on its face, absurd overparameterization — the `target` model could
plausibly memorize every reachable canonical position it will ever see.
Three considerations keep the envelope useful. First, the contract
sizes what runtimes must *accept*, and headroom is cheap insurance for
future feature channels (search-summary planes, history planes,
probe-artifact embeddings are all documented follow-ups). Second,
overparameterized-but-regularized networks trained on soft targets are
in practice strong *interpolators*, and interpolation across the book
frontier is precisely the deployment niche. Third — and most
practically — the presets make capacity an experiment, not a
commitment: if `small` matches `target` on the H2H gates, the deployed
checkpoint should be `small`, and the scaling guide says so. The honest
default position is that Quantik's ceiling is exact play; the model's
value is compute-bounded play *between* book depth and solver budget.

### 4.3 Masking at the loss versus in the architecture

Baking the mask into the network (e.g., multiplying logits by a
legality plane) couples the model artifact to rule evaluation and
breaks the contract that engines own legality. Masking at the loss
keeps the artifact rule-free but means the raw network can place mass
on illegal actions at inference; the adapter's mask makes this
harmless, and the pre-mask illegal-mass metric keeps it observable.
This mirrors standard practice in the AlphaZero line, where illegal
moves are masked out at the search boundary rather than inside the
network.

### 4.4 Book memorization versus generalization

The opening book is exact where it exists and worthless one ply beyond
its frontier. The model is approximate everywhere. The planned
engine-side composition is probe-then-evaluate: book hit → exact answer;
miss → model-guided search. The evaluation protocol is designed around
the frontier for exactly this reason: measuring model+search on
positions *inside* the book would reward memorizing labels the system
already has for free.

## 5. Planned evaluation

The acceptance gates (from the contracts model-project document) map to
four measurements:

1. **Fixed-budget search improvement.** At equal node/time budgets,
   MCTS with model priors/values versus vanilla MCTS, minimax, and
   beam, on held-out (test-shard) positions with solver references:
   move-agreement with optimal sets and value sign accuracy.
2. **Book-frontier H2H.** Round-robin from positions at and just past
   book depth: model+search versus book-only and search-only players,
   using the existing `game-result.v1` H2H machinery.
3. **Legality.** Post-mask illegal probability mass must be ~0 by
   construction; pre-mask mass is tracked as a learning diagnostic.
4. **Calibration.** Reliability of the value head against exact labels
   on the test shard, recorded in the checkpoint's training report (the
   manifest's `calibration_report` pointer).

No results are claimed in this revision; the smoke pipeline validates
plumbing (loss decreases, export validates, splits deterministic), not
strength.

## 6. AlphaZero-style distillation beyond board games

The design pattern used here — search or solver as a label factory, a
compact network as its compression, contracts as the artifact boundary
— is the same one that has traveled far outside Go and chess, which is
useful context for why a "toy" domain still merits the engineering:

- **Games as the proving ground.** Hex fell to Expert Iteration [5];
  Polygames extended zero-learning across many boards, beating strong
  humans at 19×19 Hex [8]; OpenSpiel standardizes the experimental
  substrate across dozens of games [9]. Small-board studies remain the
  standard vehicle for isolating algorithmic questions cheaply — which
  is Quantik's role here.
- **Algorithm discovery.** AlphaTensor recast matrix-multiplication
  algorithm discovery as a single-player tensor game and beat
  Strassen-era bounds for 4×4 over F2 [11]; AlphaDev did the same for
  short sorting networks at the assembly level, landing kernels in
  LLVM's libc++ [12]. Both are AlphaZero-style policy/value + MCTS on
  domains nobody would call games.
- **Engineering design.** RL chip placement [10] treats macro
  placement as a sequential game with a learned value function
  generalizing across netlists — the same "generalize past the stored
  frontier" bet this project makes past the opening book.
- **Reasoning.** HyperTree Proof Search brought the
  AlphaZero-inspired search/value loop to interactive theorem proving
  [13], and the combinatorial-optimization survey literature catalogs
  the pattern across routing, scheduling, and graph problems [14].

The through-line: whenever exact/expensive procedures (solvers, search,
compilers, provers) can label states, a policy/value network can
amortize them, and search can spend the amortized model where the
labels run out. Quantik is a minimal, fully-instrumented instance of
that loop with contract-enforced boundaries at every arrow.

## 7. Reproducibility

Every training input is a versioned artifact reproducible from the
public pipeline (`scripts/run_smoke_pipeline.sh` for the tiny corpus;
the contracts data-milestone list for the full corpus). Training is
seeded; splits are content-addressed; the exported manifest hashes the
weights and embeds dataset row counts and sources. The E2E and
train-smoke CI workflows exercise the full path from contract
validation through Rust generation to checkpoint export on every push.

## References

[1] D. Silver et al. Mastering the game of Go without human knowledge.
*Nature* 550, 354–359 (2017). https://www.nature.com/articles/nature24270

[2] D. Silver et al. Mastering Chess and Shogi by Self-Play with a
General Reinforcement Learning Algorithm. arXiv:1712.01815 (2017).
https://arxiv.org/abs/1712.01815

[3] J. Schrittwieser et al. Mastering Atari, Go, Chess and Shogi by
Planning with a Learned Model. arXiv:1911.08265 (2019).
https://arxiv.org/abs/1911.08265

[4] D. J. Wu. Accelerating Self-Play Learning in Go. arXiv:1902.10565
(2019). https://arxiv.org/abs/1902.10565

[5] T. Anthony, Z. Tian, D. Barber. Thinking Fast and Slow with Deep
Learning and Tree Search. arXiv:1705.08439 (2017).
https://arxiv.org/abs/1705.08439

[6] I. Danihelka, A. Guez, J. Schrittwieser, D. Silver. Policy
improvement by planning with Gumbel. ICLR 2022.
https://openreview.net/forum?id=bERaNdoegnO

[7] Y. Tian et al. ELF OpenGo: An Analysis and Open Reimplementation
of AlphaZero. arXiv:1902.04522 (2019). https://arxiv.org/abs/1902.04522

[8] T. Cazenave et al. Polygames: Improved Zero Learning.
arXiv:2001.09832 (2020). https://arxiv.org/abs/2001.09832

[9] M. Lanctot et al. OpenSpiel: A Framework for Reinforcement
Learning in Games. arXiv:1908.09453 (2019).
https://arxiv.org/abs/1908.09453

[10] A. Mirhoseini, A. Goldie et al. Chip Placement with Deep
Reinforcement Learning. arXiv:2004.10746 (2020).
https://arxiv.org/abs/2004.10746

[11] A. Fawzi et al. Discovering faster matrix multiplication
algorithms with reinforcement learning. *Nature* 610, 47–53 (2022).
https://www.nature.com/articles/s41586-022-05172-4

[12] D. J. Mankowitz et al. Faster sorting algorithms discovered using
deep reinforcement learning. *Nature* 618, 257–263 (2023).
https://www.nature.com/articles/s41586-023-06004-9

[13] G. Lample et al. HyperTree Proof Search for Neural Theorem
Proving. arXiv:2205.11491 (2022). https://arxiv.org/abs/2205.11491

[14] N. Mazyavkina, S. Sviridov, S. Ivanov, E. Burnaev. Reinforcement
Learning for Combinatorial Optimization: A Survey. arXiv:2003.03600
(2020). https://arxiv.org/abs/2003.03600
