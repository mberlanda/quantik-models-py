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
