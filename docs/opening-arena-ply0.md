# Opening arena from ply 0 and ply 1: measured results

This document records what the ply-0 and ply-1 arena runs measured. It contains
no interpretation. The arena was run with `scripts/evaluate_opening_arena.sh`,
which is not modified.

The distinct-games audit comes first, because every win rate below depends on it.

## 1. Distinct-games audit

Each row is one run, one start ply, and one agent family. The "worst pairing"
is the ordered pairing with the lowest distinct-games ratio, as printed by
`quantik_models.arena.autoplay` (`distinct games: worst pairing N/GAMES`).
The distinct count is the number of different action sequences among the 300
games of a pairing.

The pass rule is the threshold in `autoplay.py`, which prints its WARNING when
the worst-pairing ratio is below 0.5. No other threshold was set for this
document. The raw counts are given so another cutoff can be applied to them.

| Config | Start ply | Family | Pairings x games | Worst pairing (distinct/games) | Worst ratio | Distinct across pairings (min / median / max) | Audit |
|---|---|---|---|---|---|---|---|
| A | 0 | policy | 30 x 300 | resnet vs patience-cpool-v2: 292/300 | 97.3% | 292 / 300 / 300 | pass |
| A | 0 | MCTS (128 sims) | 42 x 300 | cpool-mcts128 vs patience-cpool-v2-mcts128: 103/300 | 34.3% | 103 / 209 / 279 | **FAIL: not reportable** |
| A | 1 | policy | 30 x 300 | patience-cpool-v2 vs mlp: 296/300 | 98.7% | 296 / 300 / 300 | pass |
| A | 1 | MCTS (128 sims) | 42 x 300 | uniform-mcts128 vs cpool-mcts128: 257/300 | 85.7% | 257 / 289 / 298 | pass |
| B (rerun, TEMP_PLIES=6) | 0 | policy | 30 x 300 | resnet vs patience-cpool-v2: 299/300 | 99.7% | 299 / 300 / 300 | pass |
| B (rerun, TEMP_PLIES=6) | 0 | MCTS (128 sims) | 42 x 300 | cpool-mcts128 vs patience-cpool-v2-mcts128: 244/300 | 81.3% | 244 / 291 / 300 | pass |

Config A, start ply 0, MCTS fails the audit. Its win rates are not reported
anywhere in this document, and its seat split is omitted too. Its data remain
on disk at `runs/eval/opening-ply0/mcts-p0/games.json`. Config B was run
because of this failure. Its MCTS cell at start ply 0 passes the same rule, as
recorded above. Config B is a separate configuration. It does not replace
Config A, and Config A's failed cell stays not reportable.

## 2. Runs

Output directories (gitignored, on the main checkout's disk):

- Config A: `/Users/mauroberlanda/Code/quantik-ns/quantik-models-py/runs/eval/opening-ply0`
  - `policy-p0`, `policy-p1`, `mcts-p0`, `mcts-p1`
  - the `p<N>` suffix is the start ply, not a seat
- Config B (rerun): `/Users/mauroberlanda/Code/quantik-ns/quantik-models-py/runs/eval/opening-ply0-tp6`
  - `policy-p0`, `mcts-p0`
  - start ply 0 only (`START_PLIES=0`)

Command for Config A. The run record gives no environment overrides, so the
script defaults apply. The console logs were lost, so this is not re-verified:

```sh
scripts/evaluate_opening_arena.sh runs/eval/opening-ply0 \
  cpool=runs/train/swept-cpool/best attn=runs/train/swept-attn/best \
  resnet=runs/train/lineup-resnet/best mlp=runs/train/lineup-mlp/best \
  patience-cpool=runs/train/patience-cpool/best \
  patience-cpool-v2=runs/train/patience-cpool-v2/best
```

Command for Config B:

```sh
START_PLIES=0 TEMP_PLIES=6 scripts/evaluate_opening_arena.sh runs/eval/opening-ply0-tp6 \
  cpool=runs/train/swept-cpool/best attn=runs/train/swept-attn/best \
  resnet=runs/train/lineup-resnet/best mlp=runs/train/lineup-mlp/best \
  patience-cpool=runs/train/patience-cpool/best \
  patience-cpool-v2=runs/train/patience-cpool-v2/best
```

Parameters, as recorded in each run's `games.json` and `agents-*.json`:

| Parameter | Config A | Config B |
|---|---|---|
| Seed | 20261001 | 20261001 |
| GAMES (per ordered pairing) | 300 | 300 |
| MCTS_SIMS | 128 | 128 |
| TEMP | 1.0 | 1.0 |
| TEMP_PLIES | 4 | 6 |
| START_PLIES | 0 and 1 | 0 |
| Policy games per start ply | 9,000 (30 pairings) | 9,000 (30 pairings) |
| MCTS games per start ply | 12,600 (42 pairings) | 12,600 (42 pairings) |

The MCTS arm includes the `uniform-mcts128` control (uniform priors, value
zero, same search settings) alongside the six network agents.

Agents, in the order the spec files list them: cpool, attn, resnet, mlp,
patience-cpool, patience-cpool-v2. Each has its own checkpoint under
`runs/train/*/best`. The checkpoints are not hashed here.

The seed is not a training seed. A search of the repo docs, scripts, src and
tests, the workspace tasks and context, and `runs/train` found no use of
20261001 as a training seed. The seed is reserved for this arena in the
workspace packets. `plan.md` also contains a GAMES=6 smoke command with this
seed. This document does not examine whether that smoke run produced output.

## 3. Leaderboards

Win rates are split by seat and are never pooled. "Mover" is the agent that
makes the first engine-chosen move from the start position, which is
`play_recorded`'s `mover`. At start ply 0 that is the first side to move from
the empty board. At start ply 1 one random ply has already been played, and
that ply is not attributed to either agent. The mover is therefore the side
that moves second in the full game. "Responder" is the other agent.

Each rate is wins over games in that seat. Denominators: a policy agent is
mover or responder in 5 opponents x 300 = 1,500 games per seat. An MCTS agent
is in 6 x 300 = 1,800 games per seat, and those include the games against
`uniform-mcts128`. Rows are in spec order, not sorted by rate. The arena's
printed leaderboard pools both seats and is not reproduced here.

Rates are computed from the `results` list in each `games.json` with the
arena's `Game` class. The stored `distinct_games` values were compared with a
recomputation for all six output directories and matched. The stored pooled
leaderboard was not compared in this pass.

### Config A, start ply 0, policy (audit: pass)

| Agent | Mover: wins/games | Mover rate | Responder: wins/games | Responder rate |
|---|---|---|---|---|
| cpool | 733/1500 | 48.9% | 613/1500 | 40.9% |
| attn | 851/1500 | 56.7% | 512/1500 | 34.1% |
| resnet | 870/1500 | 58.0% | 475/1500 | 31.7% |
| mlp | 817/1500 | 54.5% | 425/1500 | 28.3% |
| patience-cpool | 864/1500 | 57.6% | 576/1500 | 38.4% |
| patience-cpool-v2 | 1145/1500 | 76.3% | 1119/1500 | 74.6% |

### Config A, start ply 0, MCTS (audit: FAIL, not reportable)

No win rates. See section 1.

### Config A, start ply 1, policy (audit: pass)

| Agent | Mover: wins/games | Mover rate | Responder: wins/games | Responder rate |
|---|---|---|---|---|
| cpool | 611/1500 | 40.7% | 746/1500 | 49.7% |
| attn | 527/1500 | 35.1% | 809/1500 | 53.9% |
| resnet | 522/1500 | 34.8% | 873/1500 | 58.2% |
| mlp | 440/1500 | 29.3% | 831/1500 | 55.4% |
| patience-cpool | 562/1500 | 37.5% | 808/1500 | 53.9% |
| patience-cpool-v2 | 1112/1500 | 74.1% | 1159/1500 | 77.3% |

### Config A, start ply 1, MCTS 128 sims (audit: pass)

| Agent | Mover: wins/games | Mover rate | Responder: wins/games | Responder rate |
|---|---|---|---|---|
| cpool-mcts128 | 913/1800 | 50.7% | 1028/1800 | 57.1% |
| attn-mcts128 | 616/1800 | 34.2% | 1183/1800 | 65.7% |
| resnet-mcts128 | 835/1800 | 46.4% | 1316/1800 | 73.1% |
| mlp-mcts128 | 653/1800 | 36.3% | 1268/1800 | 70.4% |
| patience-cpool-mcts128 | 985/1800 | 54.7% | 1154/1800 | 64.1% |
| patience-cpool-v2-mcts128 | 1259/1800 | 69.9% | 1383/1800 | 76.8% |
| uniform-mcts128 (control) | 2/1800 | 0.1% | 5/1800 | 0.3% |

### Config B (rerun, TEMP_PLIES=6), start ply 0, policy (audit: pass)

| Agent | Mover: wins/games | Mover rate | Responder: wins/games | Responder rate |
|---|---|---|---|---|
| cpool | 754/1500 | 50.3% | 622/1500 | 41.5% |
| attn | 840/1500 | 56.0% | 576/1500 | 38.4% |
| resnet | 777/1500 | 51.8% | 478/1500 | 31.9% |
| mlp | 679/1500 | 45.3% | 406/1500 | 27.1% |
| patience-cpool | 857/1500 | 57.1% | 637/1500 | 42.5% |
| patience-cpool-v2 | 1196/1500 | 79.7% | 1178/1500 | 78.5% |

### Config B (rerun, TEMP_PLIES=6), start ply 0, MCTS 128 sims (audit: pass)

| Agent | Mover: wins/games | Mover rate | Responder: wins/games | Responder rate |
|---|---|---|---|---|
| cpool-mcts128 | 725/1800 | 40.3% | 833/1800 | 46.3% |
| attn-mcts128 | 1159/1800 | 64.4% | 737/1800 | 40.9% |
| resnet-mcts128 | 1302/1800 | 72.3% | 893/1800 | 49.6% |
| mlp-mcts128 | 1294/1800 | 71.9% | 760/1800 | 42.2% |
| patience-cpool-mcts128 | 1163/1800 | 64.6% | 957/1800 | 53.2% |
| patience-cpool-v2-mcts128 | 1413/1800 | 78.5% | 1342/1800 | 74.6% |
| uniform-mcts128 (control) | 12/1800 | 0.7% | 10/1800 | 0.6% |

## 4. Wall-clock time and provenance

- **Wall-clock time is unknown.** The console logs of both runs were lost.
  The only timestamps available are last-write times of the output files,
  which are not durations. They are given in local time for the record:
  - Config A: `agents-policy.json` last written 17:04:33; `policy-p0/games.json`
    17:06:55; `policy-p1/games.json` 17:09:06; `mcts-p0/games.json` 18:01:59;
    `mcts-p1/games.json` 18:45:39.
  - Config B: `agents-policy.json` last written 20:40:57; `policy-p0/games.json`
    20:43:09; `mcts-p0/games.json` 21:31:55.
  - The Config A policy-p0 file was written 2 min 22 s after the spec file, and
    policy-p1 2 min 11 s after policy-p0. Neither matches the roughly 35-minute
    policy-arena estimate in
    `quantik-workspace/tasks/active/QW-024-opening-arena-from-ply-zero/status.md`.
    The timestamps are therefore not used as timings.
- **Code and environment for the original runs were not recorded.** The
  commit of `quantik-models-py` and the Python environment used for Config A
  and Config B are unknown. The script's default is `PYTHON=.venv/bin/python`.
- **Recomputation code.** The recomputation imported
  `src/quantik_models/arena/autoplay.py` from the main checkout. A diff against
  the worktree copy reported the files identical.

## 5. Limits of this document

- **One training seed per checkpoint.** No margin in this document has a
  run-to-run error bar. Every checkpoint was trained once.
- **Rates are over games, not distinct games.** The distinct counts in section
  1 show how many of the 300 games in each pairing differ. No confidence
  intervals are computed here.
- **Config B is not a replacement for Config A.** It is a separate
  configuration with its own audit. Config A's MCTS cell at start ply 0 remains
  not reportable.
