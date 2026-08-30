#!/usr/bin/env bash
# Stage local checkpoints and build the public play image, so a size
# comparison between model sets is one invocation each rather than
# hand-editing a manifest.
#
#   scripts/build_docker_image.sh best   # swept-cpool only — smallest image
#   scripts/build_docker_image.sh full   # the published lineup + v3-cpool
#
# Weights come from local `runs/train/*/best` checkpoints, not the
# Hugging Face Hub — this is the local/dev build. WORKSTREAMS.md
# workstream 13 names pulling from the Hub at build time as the
# production path once the image itself is settled; nothing here does
# that yet.
#
# Only manifest.json and model.onnx are staged: the image runs
# `--runtime onnx` (see docker/Dockerfile), so weights.safetensors and
# training-report.json would be dead weight in the image for no benefit.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {best|full} [tag]" >&2
  exit 2
fi
MODE="$1"
TAG="${2:-quantik-play:$MODE}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"       # quantik-models-py
WORKSPACE="$(cd "$ROOT/.." && pwd)"            # quantik-ns

case "$MODE" in
  best)
    CHECKPOINTS="cpool=runs/train/swept-cpool/best"
    ;;
  full)
    # The published lineup (see docs/README.md) plus v3-cpool, matching
    # the five models the LAN play service already offers — see
    # play-service.md and the quantik-play-service-plan memory.
    CHECKPOINTS="cpool=runs/train/swept-cpool/best
attn=runs/train/swept-attn/best
resnet=runs/train/lineup-resnet/best
mlp=runs/train/lineup-mlp/best
v3-cpool=runs/train/v3-cpool/best"
    ;;
  *)
    echo "unknown mode $MODE; want 'best' or 'full'" >&2
    exit 2
    ;;
esac

STAGING="$ROOT/docker/staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"

while IFS='=' read -r name rel; do
  [ -z "$name" ] && continue
  src="$ROOT/$rel"
  [ -f "$src/manifest.json" ] || { echo "no such checkpoint: $src" >&2; exit 1; }
  [ -f "$src/model.onnx" ] || { echo "$src has no model.onnx" >&2; exit 1; }
  dest="$STAGING/$name"
  mkdir -p "$dest"
  cp "$src/manifest.json" "$dest/manifest.json"
  cp "$src/model.onnx" "$dest/model.onnx"
  echo "staged $name <- $rel"
done <<EOF
$CHECKPOINTS
EOF

echo
echo "building $TAG from $WORKSPACE ..."
docker build -f "$ROOT/docker/Dockerfile" -t "$TAG" "$WORKSPACE"
echo
docker images "$TAG" --format 'table {{.Repository}}:{{.Tag}}	{{.Size}}'
