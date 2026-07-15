"""Materialize Quantik contract artifacts into NumPy training views."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from quantik_core.artifact_data import (
    ObservationRow,
    load_observations_parquet,
    parse_observation_row,
)
from quantik_core.ml_data import (
    ACTION_COUNT,
    SelfPlayRow,
    load_selfplay_jsonl,
    load_selfplay_parquet,
    policy_visits_to_distribution,
    qfen_to_tensor,
)

from .labels import sample_weight

TAG_SEPARATOR = "\x1f"


@dataclass(frozen=True)
class TrainingDatasetView:
    tensors: npt.NDArray[np.float32]
    policy_target: npt.NDArray[np.float32]
    value_target: npt.NDArray[np.float32]
    sample_weight: npt.NDArray[np.float32]
    legal_action_mask: npt.NDArray[np.uint64]
    side_to_move: npt.NDArray[np.uint8]
    source_tags: tuple[tuple[str, ...], ...]

    def __len__(self) -> int:
        return int(self.value_target.shape[0])


def load_observations_jsonl(path: str | Path) -> list[ObservationRow]:
    rows: list[ObservationRow] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid observation JSON on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"observation line {line_number} must be an object")
            try:
                rows.append(parse_observation_row(record))
            except ValueError as exc:
                raise ValueError(
                    f"invalid observation row {line_number}: {exc}"
                ) from exc
    return rows


def _policy_from_dense(visits: Sequence[int]) -> npt.NDArray[np.float32]:
    if len(visits) != ACTION_COUNT:
        raise ValueError(f"policy_visits must contain {ACTION_COUNT} entries")
    array = np.asarray(visits, dtype=np.float32)
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError("policy_visits must contain at least one visit")
    return array / total


def _mask_from_policy(policy: npt.NDArray[np.float32]) -> np.uint64:
    mask = 0
    for action, probability in enumerate(policy):
        if probability > 0.0:
            mask |= 1 << action
    return np.uint64(mask)


def from_observations(rows: Iterable[ObservationRow]) -> TrainingDatasetView:
    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one observation row is required")

    tensors = np.stack(
        [qfen_to_tensor(row.qfen, row.side_to_move) for row in materialized]
    ).astype(np.float32, copy=False)
    policies = np.stack(
        [_policy_from_dense(row.policy_visits) for row in materialized]
    ).astype(np.float32, copy=False)
    values = np.asarray(
        [min(1.0, max(-1.0, row.value)) for row in materialized],
        dtype=np.float32,
    )
    weights = np.asarray(
        [sample_weight(row.value_source, row.source_confidence) for row in materialized],
        dtype=np.float32,
    )
    masks = np.asarray([row.legal_action_mask for row in materialized], dtype=np.uint64)
    sides = np.asarray([row.side_to_move for row in materialized], dtype=np.uint8)
    tags = tuple(
        (
            "schema:observation.v1",
            f"run:{row.run_id}",
            f"engine:{row.engine_kind}",
            f"value:{row.value_source}",
            (
                "policy:single-visit"
                if sum(row.policy_visits) == 1
                else "policy:visits"
            ),
        )
        for row in materialized
    )
    return TrainingDatasetView(tensors, policies, values, weights, masks, sides, tags)


def from_selfplay(rows: Iterable[SelfPlayRow]) -> TrainingDatasetView:
    materialized = list(rows)
    if not materialized:
        raise ValueError("at least one selfplay row is required")

    tensors = np.stack(
        [qfen_to_tensor(row.qfen, row.side_to_move) for row in materialized]
    ).astype(np.float32, copy=False)
    policies = np.stack(
        [policy_visits_to_distribution(row.policy) for row in materialized]
    ).astype(np.float32, copy=False)
    values = np.asarray([row.value for row in materialized], dtype=np.float32)
    weights = np.asarray(
        [sample_weight("selfplay", 1.0) for _ in materialized],
        dtype=np.float32,
    )
    masks = np.asarray([_mask_from_policy(policy) for policy in policies], dtype=np.uint64)
    sides = np.asarray([row.side_to_move for row in materialized], dtype=np.uint8)
    tags = tuple(
        ("schema:selfplay.v1", f"game:{row.game_id}", "value:selfplay")
        for row in materialized
    )
    return TrainingDatasetView(tensors, policies, values, weights, masks, sides, tags)


def write_npz(view: TrainingDatasetView, path: str | Path) -> None:
    encoded_tags = np.asarray(
        [TAG_SEPARATOR.join(tags) for tags in view.source_tags], dtype=np.str_
    )
    np.savez_compressed(
        Path(path),
        tensors=view.tensors,
        policy_target=view.policy_target,
        value_target=view.value_target,
        sample_weight=view.sample_weight,
        legal_action_mask=view.legal_action_mask,
        side_to_move=view.side_to_move,
        source_tags=encoded_tags,
    )


def load_npz(path: str | Path) -> TrainingDatasetView:
    with np.load(Path(path), allow_pickle=False) as data:
        tags = tuple(
            tuple(str(item).split(TAG_SEPARATOR))
            for item in data["source_tags"].tolist()
        )
        return TrainingDatasetView(
            tensors=data["tensors"].astype(np.float32, copy=False),
            policy_target=data["policy_target"].astype(np.float32, copy=False),
            value_target=data["value_target"].astype(np.float32, copy=False),
            sample_weight=data["sample_weight"].astype(np.float32, copy=False),
            legal_action_mask=data["legal_action_mask"].astype(np.uint64, copy=False),
            side_to_move=data["side_to_move"].astype(np.uint8, copy=False),
            source_tags=tags,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize Quantik model training data"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observations-jsonl")
    source.add_argument("--observations-parquet")
    source.add_argument("--selfplay-jsonl")
    source.add_argument("--selfplay-parquet")
    parser.add_argument("--output-npz", required=True)
    args = parser.parse_args(argv)

    if args.observations_jsonl:
        view = from_observations(load_observations_jsonl(args.observations_jsonl))
    elif args.observations_parquet:
        view = from_observations(load_observations_parquet(args.observations_parquet))
    elif args.selfplay_jsonl:
        view = from_selfplay(load_selfplay_jsonl(args.selfplay_jsonl))
    else:
        view = from_selfplay(load_selfplay_parquet(args.selfplay_parquet))

    write_npz(view, args.output_npz)
    print(f"wrote {args.output_npz} rows={len(view)} tensors={view.tensors.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
