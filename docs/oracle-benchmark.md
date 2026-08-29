# The oracle benchmark: how good is any of this?

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

Every win rate this project had published until now was measured against
another one of its own networks, or against `uniform-mcts` — a control whose
evaluator returns zero everywhere. Those answer *which of these is better*.
They cannot answer *is any of them good*, because the floor moves with the
field: a lineup of four bad models still produces a leader at 57%.

This is the fixed opponent. `quantik-core`'s alpha-beta minimax at a fixed
depth is exactly the same player in every run, on every seed, forever.

## Choosing the depth, before spending the budget

The depth was measured rather than assumed, because an earlier note in this
project put `minimax-d2` at 1.1 s a move and `minimax-d4` at 57 s — figures
taken while a twelve-run sweep and an exact solver were saturating the
machine. Re-timed on a quiet one, median seconds per move:

| start ply | `minimax-d2` | `minimax-d4` |
|---|---|---|
| 0 | 0.025 | 3.98 |
| 1 | 0.152 | 24.3 |
| 2 | 0.279 | — |
| 3 | 0.156 | 16.4 |
| 6 | 0.045 | 0.64 |

`d2` never exceeds 0.28 s anywhere, so the old 1.1 s was contention rather
than a shallower start. **A timing taken under load is an upper bound**, and
the difference here is between "not something to start alongside other work"
and one hour.

At 0.156 s a move and roughly five oracle moves per game from a ply-3 start,
1,000 games per ordered pairing costs about 17 minutes — eight pairings, an
afternoon. `d4` at 16.4 s is three orders of magnitude off that.

### The fixed-clock configuration is not usable as an oracle

The obvious alternative is a time budget rather than a depth, since that is
what `fixed_time_baselines` offers. It does not work here:

| configuration | measured, ply 3 | measured, ply 6 |
|---|---|---|
| `minimax@10ms` | 0.157 s | 0.045 s |
| `minimax@50ms` | 0.158 s | 0.178 s |
| `minimax@100ms` | 0.157 s | 0.244 s |

At a ply-3 start all three spend the same 157 ms and reach the same depth as
`-d2`. Iterative deepening cannot interrupt a level, so the clock only binds
once a level costs more than the budget — and the first level already costs
more than 100 ms there. **A budget the engine ignores is not a budget**, and
an opponent whose strength depends on the machine it ran on is not an
oracle. Fixed depth is.

## Method

Four networks against `minimax-d2`, both seats, every game recorded.

- **1,000 games per ordered pairing**, eight pairings per run: each network
  moves first 1,000 times and second 1,000 times.
- **Seeds 20260902, 20260903 and 20260904** at start ply 3, plus 20260902 at
  start ply 6. None is a training seed — training used 20260827, 20260828
  and 20260901 — so a seed-linked bias shows rather than hides.
- **`--against minimax-d2`** restricts the schedule to pairings involving
  the oracle. Twelve of the twenty ordered pairings in a full round robin
  are network-versus-network and already measured; running them again would
  spend most of the budget on nothing.

```bash
scripts/oracle_benchmark.sh runs/eval/oracle-2026-08-29 \
  cpool=runs/train/swept-cpool/best attn=runs/train/swept-attn/best \
  resnet=runs/train/lineup-resnet/best mlp=runs/train/lineup-mlp/best
```

### Not subagents

The request was to dispatch subagents. What each run needs is one
`python -m quantik_models.arena.autoplay` invocation, so a subagent per run
would have started cold to issue a shell command. The four runs went in
parallel as background processes instead: the parallelism is there, the
cold starts are not. Recorded rather than substituted silently.

## The seat is not a detail

From a ply-3 start the player to move wins most games no matter who it is.
A single pooled win rate against a fixed opponent therefore mixes the
network's strength with the first-move advantage, and the two have to be
separated before the number means anything. `arena.pack` splits them.

