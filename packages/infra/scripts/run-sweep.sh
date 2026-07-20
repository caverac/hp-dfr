#!/usr/bin/env bash
#
# Launch a matched-cost sweep as N parallel Fargate tasks, one per shard.
#
# Prerequisites: the DfrPinns-Sweep stack is deployed and AWS credentials for the
# target account are configured (each environment is a separate account, so the
# credentials pick the environment). Reads the cluster, task definition, subnets
# and security group from the stack outputs, so it hard-codes no ARNs.
#
# Usage:
#   packages/infra/scripts/run-sweep.sh <sweep> <num-shards> [run-id]
#
# Example (10 shards of the steepness sweep):
#   packages/infra/scripts/run-sweep.sh steepness 10 matched-cost-001
#
# After the tasks finish, collect and merge the shards locally:
#   aws s3 sync s3://dfr-pinns-results/<run-id>/ assets/
#   uv run hp-dfr data merge-shards --which <sweep> --num-shards <num-shards>
set -euo pipefail

SWEEP="${1:?usage: run-sweep.sh <sweep> <num-shards> [run-id]}"
NUM_SHARDS="${2:?usage: run-sweep.sh <sweep> <num-shards> [run-id]}"
RUN_ID="${3:-matched-cost-$(date +%Y%m%d-%H%M%S)}"

STACK="DfrPinnsSweep"
CONTAINER="Experiment"

get_output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}

CLUSTER=$(get_output ClusterName)
TASK_DEF=$(get_output TaskDefinitionArn)
SUBNETS=$(get_output PublicSubnetIds)
SG=$(get_output TaskSecurityGroupId)
BUCKET=$(get_output ResultsBucketName)

echo "Launching ${NUM_SHARDS} shards of the '${SWEEP}' sweep as run '${RUN_ID}'"
echo "  cluster=${CLUSTER}"

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
  aws ecs run-task \
    --cluster "$CLUSTER" \
    --task-definition "$TASK_DEF" \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG}],assignPublicIp=ENABLED}" \
    --overrides "{\"containerOverrides\":[{\"name\":\"${CONTAINER}\",\"environment\":[{\"name\":\"SWEEP\",\"value\":\"${SWEEP}\"},{\"name\":\"SHARD_INDEX\",\"value\":\"${shard}\"},{\"name\":\"NUM_SHARDS\",\"value\":\"${NUM_SHARDS}\"},{\"name\":\"RUN_ID\",\"value\":\"${RUN_ID}\"}]}]}" \
    --query "tasks[0].taskArn" --output text \
    | sed "s/^/  shard ${shard}: /"
done

echo "All shards launched. Watch progress in CloudWatch (/ecs/dfr-pinns)."
echo "When done: aws s3 sync s3://${BUCKET}/${RUN_ID}/ assets/ && \\"
echo "           uv run hp-dfr data merge-shards --which ${SWEEP} --num-shards ${NUM_SHARDS}"
