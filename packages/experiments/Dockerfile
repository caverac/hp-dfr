# hp-DFR experiment container: runs one shard of the matched-cost sweep on ECS
# Fargate. Build context is packages/experiments.
#
#   docker build -t hp-dfr-experiments packages/experiments
#   docker run --rm -e SWEEP=steepness -e NUM_SHARDS=1 hp-dfr-experiments
#
# The sweeps are CPU-only (no GPU), so the CPU-only PyTorch wheels keep the image
# small.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/

# CPU-only torch keeps the image small; the sweeps do not use a GPU. The cloud
# extra adds boto3 for the S3 result upload.
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e ".[pytorch,cloud]"

# The package computes its assets directory from the repo layout by default;
# inside the container it installs at a different depth, so point it at a writable
# directory explicitly (see hp_dfr._plotting.FIG_DIR).
RUN mkdir -p /app/assets
ENV KERAS_BACKEND=torch \
    HP_DFR_ASSETS_DIR=/app/assets \
    SWEEP=steepness \
    SHARD_INDEX=0 \
    NUM_SHARDS=1 \
    RUN_ID=local

# Runs one shard and uploads its results to S3 (see scripts/run_ecs_shard.py).
CMD ["python", "scripts/run_ecs_shard.py"]
