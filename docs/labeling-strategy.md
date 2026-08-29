# Labeling Strategy

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


Training combines several label sources. The materializer preserves source tags
and emits sample weights; downstream training may rebalance or deduplicate.

## Policy labels

- `selfplay.v1`: normalize root MCTS visits.
- `observation.v1`: normalize `policy_visits[64]`.
- Future `search-summary.v1`: use only once it has real root visits/Q-values.

Single-visit observation rows are useful smoke data but weak training labels.
They are tagged as `policy:single-visit`.

## Value labels

Preferred order:

1. exact/tablebase/opening-book values,
2. bounded solver labels,
3. strong search labels,
4. generic search labels from MCTS/minimax/beam,
5. self-play outcomes,
6. H2H calibration evidence,
7. heuristic/synthetic labels.

`game-result.v1` alone is not a supervised sample because it does not carry
per-ply board tensors. It is evaluation/calibration evidence unless joined with
position traces.
