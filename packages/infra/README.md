# DFR-PINNs Infrastructure

AWS CDK infrastructure for running DFR-PINNs experiments at scale using ECS/Fargate.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           AWS Cloud                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                         VPC                              │    │
│  │  ┌─────────────────┐     ┌─────────────────────────┐    │    │
│  │  │  Public Subnet  │     │    Private Subnet       │    │    │
│  │  │                 │     │  ┌───────────────────┐  │    │    │
│  │  │  NAT Gateway    │─────│  │  Fargate Tasks    │  │    │    │
│  │  │                 │     │  │  (Experiments)    │  │    │    │
│  │  └─────────────────┘     │  └───────────────────┘  │    │    │
│  │                          └─────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │     ECR      │  │      S3      │  │     CloudWatch       │   │
│  │  Container   │  │   Results    │  │       Logs           │   │
│  │   Images     │  │   Bucket     │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js >= 22
- Yarn >= 4 (via Corepack)

## Setup

```bash
# From repository root, install all dependencies
corepack enable
yarn install

# Bootstrap CDK (first time only)
yarn cdk bootstrap

# Synthesize CloudFormation templates
yarn cdk synth
```

## Deployment

```bash
# Deploy all stacks to dev environment
yarn cdk deploy --all

# Deploy to production
yarn cdk deploy --all --context env=prod

# Deploy specific stack
yarn cdk deploy DfrPinns-dev-Ecs
```

## Running Experiments

After deployment, you can run experiments using the AWS CLI:

```bash
# Run a DFR experiment
aws ecs run-task \
  --cluster dfr-pinns-dev \
  --task-definition <task-definition-arn> \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-ids>],securityGroups=[<sg-id>]}" \
  --overrides '{
    "containerOverrides": [{
      "name": "ExperimentContainer",
      "command": ["hp-dfr", "dfr", "run", "--problem", "sine", "--epochs", "1000"]
    }]
  }'
```

## Building and Pushing Container Image

```bash
# Navigate to experiments package
cd ../experiments

# Build container
docker build -t dfr-pinns .

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag dfr-pinns:latest <account>.dkr.ecr.us-east-1.amazonaws.com/dfr-pinns-experiments:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/dfr-pinns-experiments:latest
```

## Stacks

| Stack | Description |
|-------|-------------|
| `VpcStack` | VPC with public and private subnets |
| `EcrStack` | ECR repository for container images |
| `StorageStack` | S3 bucket for experiment results |
| `EcsStack` | ECS cluster and Fargate task definitions |

## Configuration

Edit `config/environments.ts` to customize:
- Fargate task CPU/memory
- S3 retention policies
- AWS regions

## Cleanup

```bash
# Destroy all stacks
yarn cdk destroy --all

# Note: ECR repository and S3 bucket in prod are retained by default
```

## Costs

Estimated costs (us-east-1):
- VPC NAT Gateway: ~$32/month + data transfer
- Fargate: Pay per use (no idle costs)
- S3: ~$0.023/GB/month
- CloudWatch Logs: ~$0.50/GB ingested

To minimize costs:
- Use `cdk destroy` when not running experiments
- Use smaller Fargate task sizes for development
- Enable S3 lifecycle rules (already configured)
