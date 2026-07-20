# hp-DFR experiment infrastructure

AWS CDK infrastructure for running the hp-DFR sweeps on ECS Fargate, chiefly the
**matched-cost** comparison, where plain DFR is given `2.5x` the training budget of
goal-oriented DFR so the two are compared at equal wall-clock rather than equal step
count (Section 6.4 of the manuscript). The sweep fans out into shards, one Fargate
task each, which write their results to S3; the shards are pulled down and merged
locally.

One stack, `DfrPinns-Sweep`, holds everything: an S3 results bucket and a Fargate
task definition. It runs in the account's **default VPC** (no NAT gateway) and builds
the container image as a **CDK asset** (no ECR repository) - `cdk deploy` builds
`packages/experiments` and pushes it. Each environment (`development` / `production`)
is a separate AWS account, so resources carry no environment suffix.

## Environment

The app is driven by environment variables, validated with zod (`src/env.ts`):

| Variable      | Required | Notes                         |
| ------------- | -------- | ----------------------------- |
| `ENVIRONMENT` | yes      | `development` or `production` |
| `AWS_ACCOUNT` | yes      | 12-digit target account id    |
| `AWS_REGION`  | no       | defaults to `us-east-1`       |

## Workflow

Everything below spends real (small) AWS money and requires credentials for the
target account, plus Docker running locally (the image is built at deploy time).

```bash
export ENVIRONMENT=development
export AWS_ACCOUNT=<account-id>

yarn install
yarn workspace @hp-dfr/infra cdk bootstrap   # first time per account
yarn workspace @hp-dfr/infra deploy          # builds the image and deploys

# Launch 10 shards of the steepness sweep
packages/infra/scripts/run-sweep.sh steepness 10 matched-cost-001

# Collect and merge once the tasks finish (watch CloudWatch /ecs/dfr-pinns).
# The bucket is dfr-pinns-results-<environment>; run-sweep.sh prints the exact
# sync command, reading the name from the stack's ResultsBucketName output.
aws s3 sync s3://dfr-pinns-results-development/matched-cost-001/ assets/
uv run hp-dfr data merge-shards --which steepness --num-shards 10
```

`merge-shards` writes `assets/matched_cost_steepness.json`, which the analysis reads
like the other sweep data.

## Cost and teardown

Tasks run at 2 vCPU / 4 GB. The steepness sweep is 80 runs (plain DFR at `2.5x`
budget), roughly 5-6 CPU-hours; at 10 shards that is about half an hour of wall-clock
for a few dollars of Fargate. There is no NAT gateway and no idle compute, so the
only standing cost is S3 storage (cents).

Run `yarn workspace @hp-dfr/infra destroy` when done; in `development` the results
bucket is emptied and removed with the stack, in `production` it is retained.
