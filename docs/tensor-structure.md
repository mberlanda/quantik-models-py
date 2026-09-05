# Tensor Structure

The first portable model consumes the tensor already exposed by
`quantik-core-py`:

```text
shape: (9, 4, 4)
dtype: float32
```

Channels `0..7` are shape occupancy planes, but the order is
**mover-relative, not colour-ordered**: channels `0..3` are the *side to
move*'s shapes A..D, `4..7` the opponent's, and the pairing swaps with ply
parity. Channel `8` is a full-board side-to-move plane, `float(side_to_move)`
broadcast — `0.0` when player 0 is to move, `1.0` when player 1 is to move —
in both layouts. A colour-ordered variant of these same nine planes exists
(`fastboard.to_core_tensor`, for interop) but nothing in training or serving
consumes it; every trained checkpoint expects the mover-relative layout.

Discriminating fixture: QFEN `"A.../..../..../...."` places one piece —
player 0's shape A at cell (0,0) — leaving `side_to_move == 1`. A
colour-ordered encoding puts that bit in channel 0; a mover-relative
encoding puts it in channel 4 (the opponent's shape A, since player 1 is to
move). Channel 8 is `1.0` in both. An implementation that passes on
even-ply positions but fails this one has the layouts crossed. See
`architectures.md` for the full treatment.

The policy target is a 64-slot vector using `action-index.v1`:

```text
action_index = shape * 16 + position
```

Training must always apply `legal_action_mask` outside the network before move
selection. The model may learn legality, but legality remains a rules-engine
invariant.
