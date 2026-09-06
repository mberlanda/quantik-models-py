# Serves the vendored visualizer app plus the play service behind ONE
# port, with no torch install and no data store. See ../docs/play-service.md
# ("Docker: one endpoint, no store, no torch").
#
# Weights are pulled from the published brpoplpush Hugging Face repos at
# build time — no local runs/ checkpoint, no docker/staging/ to populate
# first. The vendored app is package data (src/quantik_models/play/app/,
# see scripts/sync_visualizer.py and QW-030 M1), so this builds from
# quantik-models-py's own root; no sibling quantik-qfen-visualizer checkout
# is copied in and none is needed:
#
#   docker build -f docker/Dockerfile -t quantik-play .
#   docker run -p 8000:8000 quantik-play

# --- fetch: install [hub], pull the lineup, discard the cache -------------
FROM python:3.12-slim AS fetch
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[serve,hub]'

# A throwaway cache: HF_HOME lives only in this stage, so the download
# traffic (~7 MB/model in each of two runtimes) is paid once at build
# time and never reaches the final image — only /app/models does, via
# the --copy below, which is why --copy is not optional here. A symlink
# into this stage's cache would resolve fine right now and dangle the
# moment the next stage copies the directory without the cache behind it.
ENV HF_HOME=/tmp/hf-cache
RUN quantik-models-fetch --all --stage /app/models --copy
# hub.resolve fetches both runtimes' artifacts unconditionally (nothing
# here asks for one alone), so each staged model also carries a
# model.safetensors the onnx runtime this image runs will never open —
# ~7 MB dead weight per model, doubling the actual payload for nothing.
RUN find /app/models -name model.safetensors -delete

# --- final: only what serves a game ---------------------------------------
FROM python:3.12-slim
WORKDIR /app

# Only the [serve] extra: onnxruntime, not torch and not the onnx/onnxscript
# exporter — this image runs graphs, it does not produce them. Installing
# from pyproject + src (not a built wheel) keeps this Dockerfile in sync
# with the checked-out revision without a separate packaging step.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir '.[serve]'

# Code is MIT; the weights staged below are CC BY-NC 4.0. NOTICE says so
# and is carried into the image rather than left for a reader to infer
# from LICENSE alone, which only covers the code.
COPY docker/NOTICE ./NOTICE
LABEL org.opencontainers.image.licenses="MIT AND CC-BY-NC-4.0" \
      org.opencontainers.image.description="quantik play service — code MIT, weights CC BY-NC 4.0; see /app/NOTICE" \
      org.opencontainers.image.source="https://github.com/mberlanda/quantik-models-py"

COPY --from=fetch /app/models ./models

EXPOSE 8000

# --runtime onnx: no torch import anywhere on this path (see
# play/service.py, arena/registry.load_onnx_evaluator). --no-store: this
# is the storeless deployment QW-030 asks for — see NOTICE and
# docs/play-service.md for why. No --static override: the vendored app is
# package data now, so the default already finds it. --host/--port keep
# their argparse defaults (0.0.0.0:8000), stated here for anyone reading
# only the CMD.
CMD ["python", "-m", "quantik_models.play", \
     "--models", "models", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--runtime", "onnx", "--no-store"]
