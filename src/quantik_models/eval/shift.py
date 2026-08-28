"""Score checkpoints on positions the training distribution never covered.

The validation number a training run prints is measured on the same
distribution it trained on — plies 6 to 13, where the corpus is dense. That
is not the regime an engine playing from the opening operates in. Per
`runs/coverage.md`, plies 0-5 contain **zero** training positions and ply 6
reaches 4.44% of its 901,916 canonical live positions.

So the interesting question is not "how accurate is the network" but "how
much of its accuracy survives leaving the distribution". This module
answers that against `runs/oracle/probe-large.jsonl`: 7,800 exactly-solved
positions spanning plies 4 to 12, none of which shares a canonical key with
the training corpus.

Two conventions, both inherited from `scripts/oracle_probe.py`:

* **Accuracy is measured on positions the mover provably wins.** In a lost
  position every move loses, so nothing is being tested there — counting
  them would inflate every agent equally.
* **Value truth is the game-theoretic outcome**, +1 or -1 from the side to
  move, not a search estimate.

    python -m quantik_models.eval.shift \\
      --checkpoint runs/train/lineup-resnet/best \\
      --checkpoint runs/train/lineup-mlp/best \\
      --checkpoint runs/train/lineup-cpool/best
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..env import fastboard as fb


@dataclass
class ProbeSet:
    boards: np.ndarray  # (n, 8) uint16
    plies: np.ndarray  # (n,) int
    won: np.ndarray  # (n,) bool — mover wins with perfect play
    optimal: list[set[int]]  # outcome-preserving actions per position

    def __len__(self) -> int:
        return len(self.boards)


def load_probe(path: Path) -> ProbeSet:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    boards = np.concatenate([fb.from_qfen(row["qfen"]) for row in rows])
    return ProbeSet(
        boards=boards,
        plies=fb.popcount(fb.occupancy(boards)).astype(np.int64),
        won=np.array([bool(row["won"]) for row in rows]),
        optimal=[set(row["outcome_optimal"]) for row in rows],
    )


def assert_held_out(probe: ProbeSet, corpus_boards: np.ndarray) -> int:
    """Fail loudly if the probe is not actually held out.

    The whole evaluation rests on this. A probe that overlaps the training
    corpus measures recall and reports it as generalisation, and nothing
    downstream would look wrong.
    """
    shared = np.intersect1d(
        np.unique(fb.canonical_keys(probe.boards)),
        np.unique(fb.canonical_keys(corpus_boards)),
    )
    if shared.size:
        raise AssertionError(
            f"{shared.size} canonical keys appear in both the probe and the "
            "training corpus; this is not a shift evaluation"
        )
    return shared.size


@dataclass
class Row:
    ply: int
    won_positions: int
    correct: int
    value_abs_error: float
    value_sign_correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.won_positions if self.won_positions else float("nan")

    @property
    def value_mae(self) -> float:
        return self.value_abs_error / self.total if self.total else float("nan")

    @property
    def value_sign(self) -> float:
        return self.value_sign_correct / self.total if self.total else float("nan")


@dataclass
class Report:
    checkpoint: str
    architecture: str
    parameter_count: int
    by_ply: dict[int, Row] = field(default_factory=dict)

    def overall(self, plies: tuple[int, ...] | None = None) -> Row:
        rows = [r for p, r in self.by_ply.items() if plies is None or p in plies]
        return Row(
            ply=-1,
            won_positions=sum(r.won_positions for r in rows),
            correct=sum(r.correct for r in rows),
            value_abs_error=sum(r.value_abs_error for r in rows),
            value_sign_correct=sum(r.value_sign_correct for r in rows),
            total=sum(r.total for r in rows),
        )


def evaluate(checkpoint: Path, probe: ProbeSet, device: str = "cpu") -> Report:
    """Batched: one forward pass over the whole probe, not one per position."""
    from ..arena.registry import load_evaluator

    manifest = json.loads((checkpoint / "manifest.json").read_text())
    evaluator = load_evaluator(str(checkpoint), device)

    legal = fb.legal_masks(probe.boards)
    priors, values = evaluator(probe.boards, legal)
    chosen = priors.argmax(axis=1)

    # A masked argmax must be legal. If this ever fires, the mask is not
    # being applied and every number below is meaningless.
    if not legal[np.arange(len(chosen)), chosen].all():
        raise AssertionError("masked argmax selected an illegal action")

    truth = np.where(probe.won, 1.0, -1.0).astype(np.float32)
    report = Report(
        checkpoint=str(checkpoint),
        architecture=manifest["architecture"],
        parameter_count=manifest["parameter_count"],
    )
    for ply in sorted(set(probe.plies.tolist())):
        rows = np.flatnonzero(probe.plies == ply)
        won_rows = rows[probe.won[rows]]
        report.by_ply[int(ply)] = Row(
            ply=int(ply),
            won_positions=len(won_rows),
            correct=sum(int(chosen[i]) in probe.optimal[i] for i in won_rows),
            value_abs_error=float(np.abs(values[rows] - truth[rows]).sum()),
            value_sign_correct=int((np.sign(values[rows]) == np.sign(truth[rows])).sum()),
            total=len(rows),
        )
    return report


# Plies 0-5 hold no training positions at all, so 4 and 5 are natively out
# of distribution rather than held out; ply 6 is 4.44% covered. Reported as
# their own bucket because that is where the question lives.
SHALLOW = (4, 5, 6)


def render(reports: list[Report]) -> str:
    plies = sorted({p for r in reports for p in r.by_ply})
    lines = ["# Shift evaluation", ""]
    lines.append(
        "Exactly-solved positions holding no canonical key in common with the "
        "training corpus. Accuracy is over positions the mover provably wins."
    )
    lines.append("")
    lines.append("| model | params | shallow (4-6) | deep (7-12) | all | value MAE | value sign |")
    lines.append("|---|---|---|---|---|---|---|")
    deep = tuple(p for p in plies if p not in SHALLOW)
    for r in reports:
        s, d, a = r.overall(SHALLOW), r.overall(deep), r.overall()
        lines.append(
            f"| `{r.architecture}` | {r.parameter_count:,} | {s.accuracy:.4f} | "
            f"{d.accuracy:.4f} | {a.accuracy:.4f} | {a.value_mae:.4f} | {a.value_sign:.4f} |"
        )

    lines += ["", "## Accuracy by ply", "", "| ply | " + " | ".join(
        f"`{r.architecture}`" for r in reports) + " |"]
    lines.append("|---" * (len(reports) + 1) + "|")
    for ply in plies:
        cells = " | ".join(f"{r.by_ply[ply].accuracy:.4f}" for r in reports)
        marker = " *" if ply in SHALLOW else ""
        lines.append(f"| {ply}{marker} | {cells} |")
    lines += ["", "`*` = shallow: no training positions at plies 4-5, 4.44% at ply 6."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, default=Path("runs/oracle/probe-large.jsonl"))
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("runs/oracle/corpus/exact-sampled.npz"),
        help="checked for overlap with the probe; pass --no-verify to skip",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    probe = load_probe(args.probe)
    print(f"probe: {len(probe):,} solved positions, plies "
          f"{probe.plies.min()}-{probe.plies.max()}")

    if not args.no_verify:
        from ..data.exact_corpus import ExactCorpus

        assert_held_out(probe, ExactCorpus.load(args.corpus).boards)
        print(f"held out: 0 canonical keys shared with {args.corpus}")

    reports = [evaluate(c, probe, args.device) for c in args.checkpoint]
    print()
    print(render(reports))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                [
                    {
                        "checkpoint": r.checkpoint,
                        "architecture": r.architecture,
                        "parameter_count": r.parameter_count,
                        "by_ply": {
                            str(p): {
                                "won_positions": row.won_positions,
                                "correct": row.correct,
                                "accuracy": row.accuracy,
                                "value_mae": row.value_mae,
                                "value_sign": row.value_sign,
                                "total": row.total,
                            }
                            for p, row in sorted(r.by_ply.items())
                        },
                    }
                    for r in reports
                ],
                indent=2,
            )
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
