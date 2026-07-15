import json
from pathlib import Path

import numpy as np

from quantik_models.data.materialize import from_selfplay, load_npz, main, write_npz
from quantik_core.ml_data import load_selfplay_jsonl


def test_selfplay_materialization_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "selfplay.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema": "selfplay.v1",
                "contract_version": "1.1.0",
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
        '{"schema":"selfplay.v1","contract_version":"1.1.0","game_id":0,"ply":0,"qfen":"..../..../..../....","side_to_move":0,"policy":[{"shape":0,"position":0,"visits":1}],"value":1.0}\n',
        encoding="utf-8",
    )
    assert main(["--selfplay-jsonl", str(source), "--output-npz", str(output)]) == 0
    assert len(load_npz(output)) == 1
