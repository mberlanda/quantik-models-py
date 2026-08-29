# Tensor Structure

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


The first portable model consumes the tensor already exposed by
`quantik-core-py`:

```text
shape: (9, 4, 4)
dtype: float32
```

Channels `0..7` are player/shape occupancy planes. Channels are ordered as the
8 bitboard planes used by `bitboard.v1`. Channel `8` is a full-board
side-to-move plane filled with `0.0` for player 0 and `1.0` for player 1.

The policy target is a 64-slot vector using `action-index.v1`:

```text
action_index = shape * 16 + position
```

Training must always apply `legal_action_mask` outside the network before move
selection. The model may learn legality, but legality remains a rules-engine
invariant.
