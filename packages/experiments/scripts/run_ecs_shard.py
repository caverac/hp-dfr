"""Container entrypoint for one shard of the matched-cost sweep on ECS Fargate.

Each Fargate task runs this script with a distinct ``SHARD_INDEX``, computes its
slice of the sweep, and uploads its shard JSON to the results bucket. The shards
are pulled down and merged locally with ``hp-dfr data merge-shards``.

Configuration is read from the environment (set by the ECS task, overridable per
task at launch):

    SWEEP           which sweep to run: "steepness" or "2d"      (default steepness)
    SHARD_INDEX     zero-based shard index for this task          (default 0)
    NUM_SHARDS      total number of shards                        (default 1)
    RUN_ID          groups a run's shards under one S3 prefix     (default "local")
    RESULTS_BUCKET  S3 bucket for results; if unset, no upload    (default unset)
    BACKEND         deep-learning backend                         (default pytorch)

The upload is skipped when ``RESULTS_BUCKET`` is unset, so the script also runs
locally for testing. boto3 is imported lazily and only when an upload is needed,
so the rest of the package does not depend on it.
"""

from __future__ import annotations

import os
from pathlib import Path

from hp_dfr.data import run_2d_matched_cost_sweep
from hp_dfr.types.common import BackendType
from hp_dfr.utils import console


def _upload(path: Path, bucket: str, key: str) -> None:
    """Upload ``path`` to ``s3://bucket/key`` using task-role credentials."""
    import boto3  # lazy: only needed inside the container

    boto3.client("s3").upload_file(str(path), bucket, key)
    console.print(f"uploaded s3://{bucket}/{key}", markup=False)


def main() -> None:
    """Run this task's shard and, if configured, upload its results to S3."""
    which = os.environ.get("SWEEP", "steepness")
    shard = int(os.environ.get("SHARD_INDEX", "0"))
    num_shards = int(os.environ.get("NUM_SHARDS", "1"))
    run_id = os.environ.get("RUN_ID", "local")
    bucket = os.environ.get("RESULTS_BUCKET")
    backend: BackendType = os.environ.get("BACKEND", "pytorch")  # type: ignore[assignment]

    console.print(
        f"shard {shard}/{num_shards} of matched-cost {which} sweep (run {run_id})",
        markup=False,
    )
    out = run_2d_matched_cost_sweep(which=which, shard=shard, num_shards=num_shards, backend=backend)

    if bucket:
        _upload(out, bucket, f"{run_id}/{out.name}")
    else:
        console.print("RESULTS_BUCKET unset; skipping upload", markup=False)


if __name__ == "__main__":
    main()
