"""The constraint structure the model is wired to must be the game's.

`model.spec` restates the twelve groups so the model package stays free of
the numpy board code. That duplication is only safe if something asserts
the two agree — otherwise a change to the board geometry leaves a network
wired to a rule the engine no longer plays by, and nothing fails.
"""

from __future__ import annotations

from quantik_models.env import fastboard as fb
from quantik_models.model.spec import (
    CELL_COUNT,
    GROUP_COUNT,
    GROUP_KINDS,
    GROUPS_PER_CELL,
    constraint_groups,
)


def _cells_of(mask: int) -> tuple[int, ...]:
    return tuple(i for i in range(CELL_COUNT) if mask & (1 << i))


def test_spec_groups_are_the_engine_win_masks() -> None:
    from_spec = set(constraint_groups())
    from_engine = {_cells_of(int(m)) for m in fb.WIN_MASKS}
    assert from_spec == from_engine
    assert len(from_spec) == GROUP_COUNT


def test_every_cell_belongs_to_exactly_three_groups() -> None:
    counts = [0] * CELL_COUNT
    for cells in constraint_groups():
        for cell in cells:
            counts[cell] += 1
    assert counts == [GROUPS_PER_CELL] * CELL_COUNT


def test_group_kinds_line_up_with_the_groups() -> None:
    assert len(GROUP_KINDS) == GROUP_COUNT
    lines = [g for g, k in zip(constraint_groups(), GROUP_KINDS) if k == "line"]
    zones = [g for g, k in zip(constraint_groups(), GROUP_KINDS) if k == "zone"]
    assert len(lines) == 8 and len(zones) == 4


def test_transposing_the_board_swaps_lines_and_preserves_zones() -> None:
    """Why rows and columns share weights: transposition is in D4.

    If this ever stopped holding, tying row and column parameters would be
    an unjustified constraint rather than a symmetry the game has.
    """
    groups = constraint_groups()
    rows, cols, zones = groups[:4], groups[4:8], groups[8:]

    def transpose(cells: tuple[int, ...]) -> frozenset[int]:
        return frozenset((c % 4) * 4 + c // 4 for c in cells)

    assert {transpose(r) for r in rows} == {frozenset(c) for c in cols}
    assert {transpose(z) for z in zones} == {frozenset(z) for z in zones}
