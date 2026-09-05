# AGENTS.md

Instructions for humans and AI agents working in this repository.

## Read these first, in this order

| | |
|---|---|
| [`README.md`](README.md) | what the package is and how it is used |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | **environment, tests, CI, the release process, and the non-negotiable invariants.** Everything about working on this repository is there. |
| [`docs/README.md`](docs/README.md) | the reading order for `docs/` |

**Do not grep your way in.** This repository's hard-won conclusions are
written down, and most of them are not derivable from the code — several are
about measurements that turned out to be wrong. `DEVELOPMENT.md`'s
"Invariants that are not negotiable" is the shortest path to not repeating
one.

## What is specific to working here as an agent

**Verify before anything expensive.** Training runs and arenas cost hours,
and this repository has paid more than once for skipping the smoke test.
`DEVELOPMENT.md` has the rules; the shortest version is that a timing taken
under load is an upper bound, often a wild one, and an average from a
differently-shaped workload does not transfer.

**Say what you actually ran.** `mypy` is a gate now, but claiming a check
passed without running it has happened here. So has drawing a 16-epoch
conclusion from one epoch of evidence, and publishing a plausible, detailed,
statistically significant story that was entirely an artifact of a
hyperparameter inherited from another architecture.

**When you recommend something, write down what you rejected.** Both
architecture decision records in `docs/decisions/` are structured that way,
and the rejected options are the part that has aged best.

**A measurement disagreeing with an earlier one is the interesting case.**
Do not split the difference. Held-out accuracy has failed to predict play
strength four separate times; when it disagrees with the arena, say so.

## Where work is tracked

`quantik-workspace` is the control plane and the source of truth for what
work exists — initiatives under `tasks/active/QW-NNN/`, repository packets
under `context/repositories/`, decisions under `docs/adr/`. It is a sibling
checkout, not part of this repository.
