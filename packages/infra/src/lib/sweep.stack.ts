import * as path from "node:path";

import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Platform } from "aws-cdk-lib/aws-ecr-assets";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { type Construct } from "constructs";

import { type DeploymentEnvironment } from "@infra/env";

/** Build context for the experiment image (packages/experiments). */
const EXPERIMENTS_DIR = path.join(__dirname, "..", "..", "..", "experiments");

/** Fargate task size. The sweeps are CPU-bound; 2 vCPU / 4 GB per shard. */
const TASK_CPU = 2048;
const TASK_MEMORY_MIB = 4096;

export interface SweepStackProps extends cdk.StackProps {
  readonly environment: DeploymentEnvironment;
}

/**
 * Runs hp-DFR sweeps on Fargate and stores their results in S3.
 */
export class SweepStack extends cdk.Stack {
  public readonly resultsBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: SweepStackProps) {
    super(scope, id, props);

    this.resultsBucket = new s3.Bucket(this, "ResultsBucket", {
      bucketName: `dfr-pinns-results-${props.environment}`,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const vpc = ec2.Vpc.fromLookup(this, "DefaultVpc", { isDefault: true });

    const cluster = new ecs.Cluster(this, "Cluster", { vpc, clusterName: "dfr-pinns" });

    const logGroup = new logs.LogGroup(this, "LogGroup", {
      logGroupName: "/ecs/dfr-pinns",
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, "Task", {
      cpu: TASK_CPU,
      memoryLimitMiB: TASK_MEMORY_MIB,
      runtimePlatform: {
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
      },
    });
    this.resultsBucket.grantReadWrite(taskDefinition.taskRole);

    // The image is built from packages/experiments at deploy time; LINUX_AMD64
    // is pinned so it matches the X86_64 task even when built on an arm64 host.
    // SHARD_INDEX / NUM_SHARDS / RUN_ID are overridden per task at launch.
    taskDefinition.addContainer("Experiment", {
      image: ecs.ContainerImage.fromAsset(EXPERIMENTS_DIR, { platform: Platform.LINUX_AMD64 }),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "experiment", logGroup }),
      environment: {
        RESULTS_BUCKET: this.resultsBucket.bucketName,
        SWEEP: "steepness",
        SHARD_INDEX: "0",
        NUM_SHARDS: "1",
        RUN_ID: "manual",
        BACKEND: "pytorch",
      },
      command: ["python", "scripts/run_ecs_shard.py"],
    });

    const securityGroup = new ec2.SecurityGroup(this, "TaskSecurityGroup", {
      vpc,
      description: "hp-DFR Fargate tasks (egress only)",
      allowAllOutbound: true,
    });

    // Consumed by scripts/run-sweep.sh. Tasks launch into the default public
    // subnets with a public IP.
    new cdk.CfnOutput(this, "ClusterName", { value: cluster.clusterName });
    new cdk.CfnOutput(this, "TaskDefinitionArn", { value: taskDefinition.taskDefinitionArn });
    new cdk.CfnOutput(this, "PublicSubnetIds", {
      value: vpc.publicSubnets.map((subnet) => subnet.subnetId).join(","),
    });
    new cdk.CfnOutput(this, "TaskSecurityGroupId", { value: securityGroup.securityGroupId });
    new cdk.CfnOutput(this, "ResultsBucketName", { value: this.resultsBucket.bucketName });
  }
}
