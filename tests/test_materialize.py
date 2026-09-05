import json
from pathlib import Path

import numpy as np

from quantik_models.data.materialize import from_selfplay, load_npz, main, write_npz
from quantik_core.contracts import SUPPORTED_CONTRACTS_RELEASE
from quantik_core.ml_data import load_selfplay_jsonl


def test_selfplay_materialization_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "selfplay.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema": "selfplay.v1",
                "contract_version": SUPPORTED_CONTRACTS_RELEASE,
                "game_id": 0,
                "ply": 0,
                "qfen": "..../..../..../....",
                "side_to_move": 0,
                "policy": [
                    {"shape": 0, "position": 0, "visits": 3},
                    {"shape": 1, "position": 5, "visits": 1},
                ],
                "value": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    view = from_selfplay(load_selfplay_jsonl(source))
    assert view.tensors.shape == (1, 9, 4, 4)
    assert view.policy_target[0, 0] == 0.75
    assert view.policy_target[0, 21] == 0.25
    assert view.value_target.tolist() == [1.0]

    output = tmp_path / "view.npz"
    write_npz(view, output)
    loaded = load_npz(output)
    assert np.allclose(loaded.policy_target, view.policy_target)
    assert loaded.source_tags == view.source_tags


def test_cli_materializes_selfplay(tmp_path: Path) -> None:
    source = tmp_path / "selfplay.jsonl"
    output = tmp_path / "view.npz"
    source.write_text(
        json.dumps(
            {
                "schema": "selfplay.v1",
                "contract_version": SUPPORTED_CONTRACTS_RELEASE,
                "game_id": 0,
                "ply": 0,
                "qfen": "..../..../..../....",
                "side_to_move": 0,
                "policy": [{"shape": 0, "position": 0, "visits": 1}],
                "value": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--selfplay-jsonl", str(source), "--output-npz", str(output)]) == 0
    assert len(load_npz(output)) == 1


def _observation_row(row_id: int, qfen: str | None) -> "ObservationRow":
    from quantik_core.artifact_data import ObservationRow

    return ObservationRow(
        schema="observation.v1",
        contract_version=SUPPORTED_CONTRACTS_RELEASE,
        run_id="test",
        row_id=row_id,
        position_key="k",
        ply=0,
        side_to_move=0,
        bitboards=(0, 0, 0, 0, 0, 0, 0, 0),
        legal_action_mask=(1 << 64) - 1,
        engine_kind="test",
        engine_version="0",
        elapsed_ms=0,
        policy_visits=(1,) + (0,) * 63,
        value=0.0,
        value_source="test",
        source_confidence=1.0,
        qfen=qfen,
    )


def test_a_row_without_a_qfen_is_refused_by_index() -> None:
    """`ObservationRow.qfen` is optional in the contract.

    Without this the encoder raises a TypeError several frames down, naming
    neither the offending row nor the file it came from — and the index of
    the row that produced it is the one thing needed to fix the input.
    """
    import pytest

    from quantik_models.data.materialize import from_observations

    rows = [
        _observation_row(0, "..../..../..../...."),
        _observation_row(1, None),
    ]
    with pytest.raises(ValueError, match="observation row 1 has no qfen"):
        from_observations(rows)
