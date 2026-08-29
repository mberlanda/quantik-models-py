#!/usr/bin/env bash
# Stage every model in the lineup as a Hugging Face repository, in one pass.
#
# One pass, not four invocations, because the family has to move together.
# Each model gets its own Hub repo — `model-index` is per repository, so four
# models behind one card means one set of searchable metrics, wrong for three
# of them — and the cost of that split is four tags to keep in lockstep
# against a workspace that versions its components with a single number.
# Scripting the pass is what keeps the four from drifting apart by hand.
#
#   scripts/stage_hub_repos.sh staging \
#     runs/train/swept-cpool/best runs/train/swept-attn/best \
#     runs/train/lineup-resnet/best runs/train/lineup-mlp/best
#
# Writes files. Uploads nothing.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 OUT_DIR CHECKPOINT [CHECKPOINT ...]" >&2
  exit 2
fi

OUT="$1"; shift
PYTHON="${PYTHON:-.venv/bin/python}"
# Both default inside the module; named here so a run is self-describing.
NAMESPACE="${QUANTIK_HF_NAMESPACE:-brpoplpush}"
LICENSE="${HF_LICENSE:-apache-2.0}"
# The evaluation the model-index numbers are read from. Pointing this at a
# stale directory is the one way to publish a card that describes a
# checkpoint other than the one beside it.
EVAL="${EVAL_DIR:-runs/eval/swept-2026-08-30}"

for required in "$EVAL/shift.json" "$EVAL/policy-p3/games.json"; do
  [ -f "$required" ] || { echo "no such evaluation artifact: $required" >&2; exit 1; }
done

mkdir -p "$OUT"
staged=()

for checkpoint in "$@"; do
  [ -d "$checkpoint" ] || { echo "no such checkpoint: $checkpoint" >&2; exit 1; }
  arch=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['architecture'])" \
    "$checkpoint/manifest.json")
  name=$("$PYTHON" -c "from quantik_models.export.huggingface import repo_name_for; \
    print(repo_name_for('$arch'))")

  echo "== $arch -> $NAMESPACE/$name =="
  "$PYTHON" -m quantik_models.export.huggingface "$checkpoint" "$OUT/$name" \
    --namespace "$NAMESPACE" --license "$LICENSE" \
    --shift "$EVAL/shift.json" --arena "$EVAL/policy-p3/games.json" \
    --agent "${arch%%-*}"
  staged+=("$OUT/$name")
done

echo
echo "staged ${#staged[@]} repositories under $OUT:"
for dir in "${staged[@]}"; do echo "  $NAMESPACE/$(basename "$dir")"; done
echo
echo "Nothing has been uploaded. To publish one, from inside its directory:"
echo "  hf auth login"
echo "  hf upload $NAMESPACE/<name> . --repo-type model"
echo "See docs/publishing-to-hugging-face.md before the first push — the"
echo ".gitattributes mistake is the one a later commit cannot fix."
