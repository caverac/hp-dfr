#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { VpcStack } from "../lib/stacks/vpc-stack";
import { EcrStack } from "../lib/stacks/ecr-stack";
import { EcsStack } from "../lib/stacks/ecs-stack";
import { StorageStack } from "../lib/stacks/storage-stack";
import { getConfig } from "../config/environments";

const app = new cdk.App();

// Get environment from context or default to dev
const envName = app.node.tryGetContext("env") || "dev";
const config = getConfig(envName);

// Common environment settings
const env: cdk.Environment = {
  account: config.account || process.env.CDK_DEFAULT_ACCOUNT,
  region: config.region || process.env.CDK_DEFAULT_REGION,
};

// Stack naming prefix
const prefix = `DfrPinns-${config.name}`;

// VPC Stack - Network infrastructure
const vpcStack = new VpcStack(app, `${prefix}-Vpc`, {
  env,
  description: "VPC and networking for DFR-PINNs experiments",
});

// ECR Stack - Container registry
const ecrStack = new EcrStack(app, `${prefix}-Ecr`, {
  env,
  description: "ECR repository for experiment containers",
});

// Storage Stack - S3 for results
const storageStack = new StorageStack(app, `${prefix}-Storage`, {
  env,
  config,
  description: "S3 storage for experiment results",
});

// ECS Stack - Fargate cluster and tasks
const ecsStack = new EcsStack(app, `${prefix}-Ecs`, {
  env,
  vpc: vpcStack.vpc,
  repository: ecrStack.repository,
  resultsBucket: storageStack.resultsBucket,
  config,
  description: "ECS cluster and Fargate tasks for experiments",
});

// Add dependencies
ecsStack.addDependency(vpcStack);
ecsStack.addDependency(ecrStack);
ecsStack.addDependency(storageStack);

// Tags for all resources
cdk.Tags.of(app).add("Project", "DfrPinns");
cdk.Tags.of(app).add("Environment", config.name);
cdk.Tags.of(app).add("ManagedBy", "CDK");
