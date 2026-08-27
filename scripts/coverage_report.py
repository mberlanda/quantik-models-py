#!/usr/bin/env python
"""How much of Quantik did the model actually see, and is the evaluation big enough?

Answers two questions a win rate cannot:

1. **Coverage.** For every ply, how many canonical positions exist, how many
   the model was trained on, and how many the evaluation probe holds. A
   coverage number without its denominator is not a number.
2. **Power.** Whether the accuracy gap between two agents survives a *paired*
   test on the probe, per ply and pooled — since both agents face identical
   positions, an unpaired interval throws away most of the evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantik_models.arena.probe import mcnemar, paired_difference_ci, score
from quantik_models.data.exact_corpus import ExactCorpus
from quantik_models.env import fastboard as fb


def load_probe(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def probe_plies(probe: list[dict]) -> np.ndarray:
    boards = np.concatenate([fb.from_qfen(r["qfen"]) for r in probe])
    return fb.popcount(fb.occupancy(boards))


def coverage_rows(counts: dict, corpus: ExactCorpus, probes: dict[str, list[dict]]) -> list[dict]:
    policy = corpus.optimal_mask != 0
    probe_counts = {name: probe_plies(p) for name, p in probes.items()}
    plies = sorted(
        {int(k) for k in counts}
        | set(corpus.plies.tolist())
        | {int(p) for arr in probe_counts.values() for p in arr.tolist()}
    )
    rows = []
    for ply in plies:
        total = counts.get(str(ply), {}).get("live")
        mask = corpus.plies == ply
        row = {
            "ply": ply,
            "canonical_live": total,
            "trained_total": int(mask.sum()),
            "trained_policy": int((mask & policy).sum()),
            "coverage": (int(mask.sum()) / total) if total else None,
            "policy_coverage": (int((mask & policy).sum()) / total) if total else None,
        }
        for name, arr in probe_counts.items():
            row[f"probe_{name}"] = int((arr == ply).sum())
        rows.append(row)
    return rows


def fmt_pct(value, places=2):
    if value is None:
        return "—"
    if value == 0:
        return "0"
    if value < 0.0001:
        return f"{value * 100:.4f}%"
    return f"{value * 100:.{places}f}%"


def render(report: dict) -> str:
    lines = [
        "# Coverage and statistical power",
        "",
        f"Generated `{report['generated_at']}`.",
        "",
        "## How much of the game the model saw",
        "",
        "Positions are counted **up to symmetry**: Quantik is invariant under 8 board",
        "symmetries composed with 24 shape relabelings, so a position and its 191 images",
        "are one game. `canonical live` excludes terminal positions, which need no decision.",
        "",
        "| ply | canonical live | trained on | with policy label | coverage | held-out probe |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["coverage"]:
        total = f"{row['canonical_live']:,}" if row["canonical_live"] else "—"
        probe = sum(v for k, v in row.items() if k.startswith("probe_"))
        lines.append(
            f"| {row['ply']} | {total} | {row['trained_total']:,} | {row['trained_policy']:,} "
            f"| {fmt_pct(row['coverage'])} | {probe:,} |"
        )
    lines += ["", "## Is the evaluation large enough?", ""]
    for block in report["comparisons"]:
        m, ci = block["mcnemar"], block["difference"]
        lines += [
            f"### {block['a']} vs {block['b']} — {block['scope']}",
            "",
            f"- positions compared (mover provably wins): **{m['positions']:,}**",
            f"- accuracy: {m['accuracy_a']:.2%} vs {m['accuracy_b']:.2%}",
            f"- disagreements: {m['a_right_b_wrong']} where only `{block['a']}` is right, "
            f"{m['b_right_a_wrong']} where only `{block['b']}` is right",
            f"- **paired exact test p = {m['p_value']:.3g}**",
            f"- accuracy difference {ci['point']:+.2%} (95% CI {ci['low']:+.2%} to {ci['high']:+.2%})",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, default=Path("runs/canonical/counts.json"))
    parser.add_argument("--corpus", type=Path, default=Path("runs/oracle/corpus/sampled.npz"))
    parser.add_argument("--probes", type=Path, nargs="+",
                        default=[Path("runs/oracle/probe.jsonl"), Path("runs/oracle/probe-large.jsonl")])
    parser.add_argument("--agents", type=Path, default=Path("runs/oracle/showdown-final.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/coverage"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    counts = json.loads(args.counts.read_text()) if args.counts.exists() else {}
    corpus = ExactCorpus.load(args.corpus)
    probes = {p.stem: load_probe(p) for p in args.probes if p.exists()}
    combined = [row for p in probes.values() for row in p]
    print(f"probe: {len(combined):,} solved positions across {len(probes)} files", flush=True)

    specs = json.loads(args.agents.read_text())
    results = []
    for spec in specs:
        r = score(spec, combined, workers=args.workers)
        results.append(r)
        print(f"  {r.agent}: {r.outcome_accuracy:.2%} ({r.seconds:.0f}s)", flush=True)

    comparisons = []
    if len(results) >= 2:
        a, b = results[0], results[1]
        for scope, plies in (("all plies", None), ("opening (plies 4-7)", (4, 5, 6, 7))):
            point, low, high = paired_difference_ci(a, b, plies=plies)
            comparisons.append({
                "a": a.agent, "b": b.agent, "scope": scope,
                "mcnemar": mcnemar(a, b, plies=plies),
                "difference": {"point": point, "low": low, "high": high},
            })

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "coverage": coverage_rows(counts, corpus, probes),
        "agents": [
            {"agent": r.agent, "outcome_accuracy": r.outcome_accuracy,
             "by_ply": {str(k): {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]}
                        for k, v in r.by_ply().items()}}
            for r in results
        ],
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    args.out.with_suffix(".md").write_text(render(report))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
