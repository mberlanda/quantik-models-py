"""Vectorized Quantik rules over batches of positions.

The reference rules live in `quantik_core` (tuple bitboards, one position
per call). Self-play needs the same rules applied to hundreds of games at
once, so this module re-expresses them as NumPy array ops over an
`(n, 8) uint16` bitboard batch — the same channel order as
`quantik_core.commons.Bitboard`: `[p0s0, p0s1, p0s2, p0s3, p1s0, ..., p1s3]`.

`tests/test_fastboard.py` cross-checks every primitive here against
`quantik_core` on random playouts, so this stays a re-expression rather
than a second source of truth.

Action index is the shared 64-slot `shape * 16 + position` convention from
`quantik_core.ml_data`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

ACTION_COUNT = 64
BOARD_SIZE = 4
SQUARES = 16
SHAPES = 4
MAX_PIECES_PER_SHAPE = 2

_ROW_MASKS = [0b0000000000001111, 0b0000000011110000, 0b0000111100000000, 0b1111000000000000]
_COL_MASKS = [0b0001000100010001, 0b0010001000100010, 0b0100010001000100, 0b1000100010001000]
_ZONE_MASKS = [0b0000000000110011, 0b0000000011001100, 0b0011001100000000, 0b1100110000000000]
WIN_MASKS = np.array(_ROW_MASKS + _COL_MASKS + _ZONE_MASKS, dtype=np.uint16)

# REGION[pos] = row | column | 2x2 zone containing `pos`. A shape may not be
# placed on `pos` when the opponent already has that shape anywhere in it.
REGION = np.zeros(SQUARES, dtype=np.uint16)
for _pos in range(SQUARES):
    _row, _col = divmod(_pos, BOARD_SIZE)
    _zone = (_row // 2) * 2 + (_col // 2)
    REGION[_pos] = np.uint16(_ROW_MASKS[_row] | _COL_MASKS[_col] | _ZONE_MASKS[_zone])

# Bit `pos` of a uint16 board word.
SQUARE_BITS = (np.uint16(1) << np.arange(SQUARES, dtype=np.uint16)).astype(np.uint16)

# popcount over the whole uint16 domain: one 64 KiB table beats any
# bit-twiddling sequence here and keeps the hot paths a single fancy-index.
_POPCOUNT16 = np.zeros(1 << 16, dtype=np.uint8)
for _i in range(1, 1 << 16):
    _POPCOUNT16[_i] = _POPCOUNT16[_i >> 1] + (_i & 1)

# Action a = shape * 16 + position.
ACTION_SHAPE = np.repeat(np.arange(SHAPES, dtype=np.int64), SQUARES)
ACTION_POSITION = np.tile(np.arange(SQUARES, dtype=np.int64), SHAPES)


def popcount(words: npt.NDArray[np.uint16]) -> npt.NDArray[np.int64]:
    """Bit count of each uint16 entry."""
    return _POPCOUNT16[words].astype(np.int64)


def empty_boards(n: int) -> npt.NDArray[np.uint16]:
    """`n` empty 4x4 boards."""
    return np.zeros((n, 8), dtype=np.uint16)


def occupancy(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """Union of all eight piece boards."""
    return np.bitwise_or.reduce(bb, axis=1)


def side_to_move(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.int64]:
    """0 or 1 per board. Player 0 moves first and the players alternate, so
    the side to move is the parity of the number of pieces on the board."""
    return (popcount(occupancy(bb)) & 1).astype(np.int64)


def shape_unions(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    """`(n, 4)` — squares holding each shape, either color."""
    return (bb[:, :SHAPES] | bb[:, SHAPES:]).astype(np.uint16)


def has_winning_line(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.bool_]:
    """True where some row, column, or zone holds all four shapes."""
    union = shape_unions(bb)  # (n, 4)
    # (n, 4 shapes, 12 lines) -> a line wins when every shape is present.
    present = (union[:, :, None] & WIN_MASKS[None, None, :]) != 0
    return present.all(axis=1).any(axis=1)


def legal_masks(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.bool_]:
    """`(n, 64)` boolean legality for the side to move on each board.

    A placement of `shape` on `position` by `player` is legal when the
    square is empty, the player has fewer than two of that shape left, and
    the opponent has no piece of that shape in the square's row, column, or
    2x2 zone.
    """
    n = bb.shape[0]
    player = side_to_move(bb)
    rows = np.arange(n)

    own = bb[rows[:, None], (player[:, None] * SHAPES) + np.arange(SHAPES)]
    opp = bb[rows[:, None], ((1 - player)[:, None] * SHAPES) + np.arange(SHAPES)]

    # (n, 4): shapes the mover still has in hand.
    in_hand = popcount(own.astype(np.uint16)) < MAX_PIECES_PER_SHAPE
    # (n, 4, 16): opponent already holds this shape in the square's region.
    blocked = (opp[:, :, None] & REGION[None, None, :]) != 0
    # (n, 1, 16): square is free.
    free = (occupancy(bb)[:, None, None] & SQUARE_BITS[None, None, :]) == 0

    legal = in_hand[:, :, None] & ~blocked & free
    return legal.reshape(n, ACTION_COUNT)


def apply_actions(
    bb: npt.NDArray[np.uint16], actions: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint16]:
    """Place one piece per board; returns a new array (inputs untouched)."""
    out = bb.copy()
    n = bb.shape[0]
    player = side_to_move(bb)
    channel = player * SHAPES + ACTION_SHAPE[actions]
    out[np.arange(n), channel] |= SQUARE_BITS[ACTION_POSITION[actions]]
    return out


def terminal_status(
    bb: npt.NDArray[np.uint16],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """`(is_terminal, value_for_side_to_move)`.

    Both Quantik terminal conditions are losses for the side to move: the
    previous mover either completed a line, or left the mover with no legal
    reply. So a terminal value is always -1.0.
    """
    won = has_winning_line(bb)
    stuck = ~legal_masks(bb).any(axis=1)
    done = won | stuck
    return done, np.where(done, np.float32(-1.0), np.float32(0.0)).astype(np.float32)


def encode_tensors(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """`(n, 9, 4, 4)` float32 matching `tensor-board.v1`.

    Channels 0-3 are the side-to-move's shapes A-D, channels 4-7 the
    opponent's, channel 8 is the side-to-move flag broadcast over the board.

    NOTE: `quantik_core.ml_data.qfen_to_tensor` orders channels by *color*
    (player 0 first) rather than by side to move. Both are valid encodings
    of `tensor-board.v1`'s 9x4x4 shape; this one is
    perspective-relative, which is what makes a single value head with a
    side-to-move sign convention learnable. `mover_relative=False` in
    `to_core_tensor` reproduces the color-ordered layout for interop.
    """
    n = bb.shape[0]
    player = side_to_move(bb)
    rows = np.arange(n)
    order = np.concatenate(
        [
            (player[:, None] * SHAPES) + np.arange(SHAPES),
            ((1 - player)[:, None] * SHAPES) + np.arange(SHAPES),
        ],
        axis=1,
    )
    planes = bb[rows[:, None], order]  # (n, 8) uint16, mover-relative
    bits = ((planes[:, :, None] & SQUARE_BITS[None, None, :]) != 0).astype(np.float32)
    out = np.zeros((n, 9, SQUARES), dtype=np.float32)
    out[:, :8, :] = bits
    out[:, 8, :] = player[:, None].astype(np.float32)
    return out.reshape(n, 9, BOARD_SIZE, BOARD_SIZE)


def to_core_tensor(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.float32]:
    """Color-ordered `(n, 9, 4, 4)` encoding, matching `qfen_to_tensor`."""
    n = bb.shape[0]
    bits = ((bb[:, :, None] & SQUARE_BITS[None, None, :]) != 0).astype(np.float32)
    out = np.zeros((n, 9, SQUARES), dtype=np.float32)
    out[:, :8, :] = bits
    out[:, 8, :] = side_to_move(bb)[:, None].astype(np.float32)
    return out.reshape(n, 9, BOARD_SIZE, BOARD_SIZE)


_QFEN_CHARS = "ABCD"


def to_qfen(board: npt.NDArray[np.uint16]) -> str:
    """Single-board QFEN: uppercase = player 0, lowercase = player 1."""
    cells = ["."] * SQUARES
    for channel in range(8):
        word = int(board[channel])
        letter = _QFEN_CHARS[channel % SHAPES]
        if channel >= SHAPES:
            letter = letter.lower()
        for pos in range(SQUARES):
            if word & (1 << pos):
                cells[pos] = letter
    return "/".join("".join(cells[r * 4 : r * 4 + 4]) for r in range(BOARD_SIZE))


def from_qfen(qfen: str) -> npt.NDArray[np.uint16]:
    """Parse a QFEN into a `(1, 8) uint16` batch of one board."""
    rows = qfen.split("/")
    if len(rows) != BOARD_SIZE or any(len(r) != BOARD_SIZE for r in rows):
        raise ValueError(f"malformed QFEN: {qfen!r}")
    out = np.zeros((1, 8), dtype=np.uint16)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            shape = _QFEN_CHARS.find(ch.upper())
            if shape < 0:
                raise ValueError(f"bad QFEN character {ch!r}")
            channel = shape + (SHAPES if ch.islower() else 0)
            out[0, channel] |= np.uint16(1 << (r * BOARD_SIZE + c))
    return out


@dataclass
class BatchBoard:
    """Mutable batch of positions with the convenience wrappers self-play uses."""

    bb: npt.NDArray[np.uint16]

    @classmethod
    def empty(cls, n: int) -> "BatchBoard":
        return cls(empty_boards(n))

    def __len__(self) -> int:
        return int(self.bb.shape[0])

    @property
    def side_to_move(self) -> npt.NDArray[np.int64]:
        return side_to_move(self.bb)

    def legal(self) -> npt.NDArray[np.bool_]:
        return legal_masks(self.bb)

    def tensors(self) -> npt.NDArray[np.float32]:
        return encode_tensors(self.bb)

    def status(self) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
        return terminal_status(self.bb)

    def step(self, actions: npt.NDArray[np.int64]) -> "BatchBoard":
        return BatchBoard(apply_actions(self.bb, actions))

    def select(self, index: npt.NDArray[np.int64]) -> "BatchBoard":
        return BatchBoard(self.bb[index])


# --- symmetry ------------------------------------------------------------
#
# Quantik's rules are invariant under 8 spatial symmetries of the 4x4 board
# *composed with* the 24 permutations of the shape labels — 192 in all, the
# same group `quantik_core.symmetry` canonicalizes over. A position and its
# image are the same game, so a self-play row can be replayed under any of
# them: 192x augmentation for free.
#
# Not every D4 element preserves the 2x2 zone partition on its own, but all
# eight do here: the zone grid is symmetric under the full dihedral group of
# the square, and rows/columns map to rows/columns.

_IDENTITY = np.arange(SQUARES, dtype=np.int64)


def _spatial_permutations() -> npt.NDArray[np.int64]:
    """`(8, 16)` — `perm[d, src] = dst` for each dihedral element."""
    grid = _IDENTITY.reshape(BOARD_SIZE, BOARD_SIZE)
    variants = []
    for flip in (False, True):
        base = np.fliplr(grid) if flip else grid
        for turns in range(4):
            variants.append(np.rot90(base, turns))
    perms = np.zeros((8, SQUARES), dtype=np.int64)
    for d, variant in enumerate(variants):
        # variant[dst] names the source square that lands on dst.
        perms[d, variant.reshape(-1)] = _IDENTITY
    return perms


SPATIAL_PERMS = _spatial_permutations()

# One 8 x 65536 uint16 table (1 MiB) maps a whole board word through a
# dihedral element in a single fancy-index, which beats per-bit shuffling by
# orders of magnitude in the self-play hot path.
_SPATIAL_WORD_TABLE = np.zeros((8, 1 << 16), dtype=np.uint16)
for _d in range(8):
    _bit_images = (np.uint16(1) << SPATIAL_PERMS[_d].astype(np.uint16)).astype(np.uint16)
    for _w in range(1 << 16):
        _acc = 0
        _rest = _w
        while _rest:
            _low = _rest & -_rest
            _acc |= int(_bit_images[_low.bit_length() - 1])
            _rest ^= _low
        _SPATIAL_WORD_TABLE[_d, _w] = _acc

from itertools import permutations as _permutations  # noqa: E402

SHAPE_PERMS = np.array(list(_permutations(range(SHAPES))), dtype=np.int64)  # (24, 4)
SYMMETRY_COUNT = 8 * len(SHAPE_PERMS)


def transform_boards(
    bb: npt.NDArray[np.uint16],
    spatial: npt.NDArray[np.int64],
    shape: npt.NDArray[np.int64],
) -> npt.NDArray[np.uint16]:
    """Apply a per-board `(spatial, shape)` symmetry to a batch.

    `spatial[i]` indexes `SPATIAL_PERMS`, `shape[i]` indexes `SHAPE_PERMS`.
    Colors are never swapped, so the side to move is preserved.
    """
    moved = _SPATIAL_WORD_TABLE[spatial[:, None], bb]  # (n, 8)
    order = SHAPE_PERMS[shape]  # (n, 4) — order[i, new_shape] = old_shape
    both = np.concatenate([order, order + SHAPES], axis=1)  # (n, 8)
    rows = np.arange(bb.shape[0])[:, None]
    return moved[rows, both]


def transform_actions(
    actions: npt.NDArray[np.int64],
    spatial: npt.NDArray[np.int64],
    shape: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Map action indices through the same symmetry as `transform_boards`."""
    inverse_shape = np.argsort(SHAPE_PERMS[shape], axis=1)  # old_shape -> new_shape
    rows = np.arange(actions.shape[0])
    old_shape, old_pos = np.divmod(actions, SQUARES)
    new_shape = inverse_shape[rows, old_shape]
    new_pos = SPATIAL_PERMS[spatial, old_pos]
    return new_shape * SQUARES + new_pos


def transform_policies(
    policies: npt.NDArray[np.float32],
    spatial: npt.NDArray[np.int64],
    shape: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Permute `(n, 64)` action distributions through a symmetry."""
    n = policies.shape[0]
    inverse_shape = np.argsort(SHAPE_PERMS[shape], axis=1)  # (n, 4)
    # destination index for every (shape, position) slot
    dst_shape = np.repeat(inverse_shape, SQUARES, axis=1)  # (n, 64)
    dst_pos = np.tile(SPATIAL_PERMS[spatial], (1, SHAPES))  # (n, 64)
    dst = dst_shape * SQUARES + dst_pos
    out = np.zeros_like(policies)
    np.put_along_axis(out, dst, policies, axis=1)
    return out


def random_symmetries(
    n: int, rng: np.random.Generator
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Draw `n` independent `(spatial, shape)` symmetry indices."""
    return rng.integers(0, 8, size=n), rng.integers(0, len(SHAPE_PERMS), size=n)


# Every square holds at most one of 8 piece kinds, so a whole board packs
# into 16 nibbles — one uint64 — which is what makes "min over 192
# symmetries" a plain vectorized reduction.
_NIBBLES = (np.uint64(1) << (np.uint64(4) * np.arange(SQUARES, dtype=np.uint64))).astype(np.uint64)


def board_codes(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint64]:
    """Pack each board into one uint64: nibble `pos` = channel + 1, 0 = empty."""
    code = np.zeros(bb.shape[0], dtype=np.uint64)
    for channel in range(8):
        occupied = (bb[:, channel][:, None] & SQUARE_BITS[None, :]) != 0
        code += (occupied * np.uint64(channel + 1) * _NIBBLES[None, :]).sum(
            axis=1, dtype=np.uint64
        )
    return code


def canonical_keys(bb: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint64]:
    """Smallest packed code over all 192 symmetries — one key per position.

    Two positions share a key exactly when they are the same game up to
    symmetry, which is what dedup and replay-buffer keying want. Colors are
    never permuted, so the side to move is part of the identity.
    """
    best = np.full(bb.shape[0], np.iinfo(np.uint64).max, dtype=np.uint64)
    for d in range(8):
        moved = _SPATIAL_WORD_TABLE[d, bb]
        for perm in SHAPE_PERMS:
            both = np.concatenate([perm, perm + SHAPES])
            best = np.minimum(best, board_codes(moved[:, both]))
    return best
