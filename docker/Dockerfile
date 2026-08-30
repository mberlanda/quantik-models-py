# Serves the visualizer plus the play service behind ONE port, with no
# torch install and no data store. See ../docs/play-service.md ("Reaching
# it off your LAN") and WORKSTREAMS.md workstream 13 in the workspace root.
#
# Build from the quantik-ns workspace root (context has to reach both
# sibling repos):
#
#   scripts/build_docker_image.sh best   # swept-cpool only
#   scripts/build_docker_image.sh full   # the published lineup + v3-cpool
#
# That script stages the chosen checkpoints into docker/staging/ before
# calling `docker build` — this Dockerfile bakes in whatever it finds
# there and nothing else. It never reaches onto the network for weights;
# see the script for the local-checkpoint vs. Hugging-Face-Hub tradeoff.
FROM python:3.12-slim

WORKDIR /app

# Only the [serve] extra: onnxruntime, not torch and not the onnx/onnxscript
# exporter — this image runs graphs, it does not produce them. Installing
# from pyproject + src (not the built sdist) keeps this Dockerfile in sync
# with the checked-out revision without a separate packaging step.
COPY quantik-models-py/pyproject.toml quantik-models-py/README.md quantik-models-py/LICENSE ./
COPY quantik-models-py/src ./src
RUN pip install --no-cache-dir '.[serve]'

COPY quantik-models-py/docker/NOTICE ./NOTICE
COPY quantik-models-py/docker/staging ./staging
COPY quantik-qfen-visualizer ./app

EXPOSE 8000

# --runtime onnx: no torch import anywhere on this path (see
# play/service.py, arena/registry.load_onnx_evaluator). --no-store: this
# is the storeless deployment workstream 13 asks for — see NOTICE and
# docs/play-service.md for why. --host/--port keep their argparse
# defaults (0.0.0.0:8000), stated here for anyone reading only the CMD.
CMD ["python", "-m", "quantik_models.play", \
     "--models", "staging", "--static", "app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--runtime", "onnx", "--no-store"]
