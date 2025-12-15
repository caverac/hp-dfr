import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { EnvironmentConfig } from "../../config/environments";

export interface EcsStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  repository: ecr.Repository;
  resultsBucket: s3.Bucket;
  config: EnvironmentConfig;
}

/**
 * ECS Stack for running DFR-PINNs experiments.
 *
 * Creates an ECS cluster with Fargate capacity providers and
 * task definitions for running experiments.
 */
export class EcsStack extends cdk.Stack {
  /** ECS cluster for experiments */
  public readonly cluster: ecs.Cluster;
  /** Task definition for experiments */
  public readonly taskDefinition: ecs.FargateTaskDefinition;

  constructor(scope: Construct, id: string, props: EcsStackProps) {
    super(scope, id, props);

    const { vpc, repository, resultsBucket, config } = props;

    // Create ECS Cluster
    this.cluster = new ecs.Cluster(this, "ExperimentCluster", {
      vpc,
      clusterName: `dfr-pinns-${config.name}`,
      containerInsights: true,
      enableFargateCapacityProviders: true,
    });

    // Create CloudWatch Log Group
    const logGroup = new logs.LogGroup(this, "TaskLogGroup", {
      logGroupName: `/ecs/dfr-pinns-${config.name}`,
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Create Task Execution Role
    const executionRole = new iam.Role(this, "TaskExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy"
        ),
      ],
    });

    // Create Task Role (for container to access AWS services)
    const taskRole = new iam.Role(this, "TaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });

    // Grant S3 access for results
    resultsBucket.grantReadWrite(taskRole);

    // Create Fargate Task Definition
    this.taskDefinition = new ecs.FargateTaskDefinition(
      this,
      "ExperimentTask",
      {
        memoryLimitMiB: config.fargate.memoryMiB,
        cpu: config.fargate.cpu,
        executionRole,
        taskRole,
        runtimePlatform: {
          cpuArchitecture: ecs.CpuArchitecture.X86_64,
          operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
        },
      }
    );

    // Add container to task definition
    const container = this.taskDefinition.addContainer("ExperimentContainer", {
      image: ecs.ContainerImage.fromEcrRepository(repository, "latest"),
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "experiment",
        logGroup,
      }),
      environment: {
        RESULTS_BUCKET: resultsBucket.bucketName,
        AWS_REGION: this.region,
        ENVIRONMENT: config.name,
      },
      // Command can be overridden when running the task
      command: ["dfr-pinns", "--help"],
    });

    // Security group for tasks
    const taskSecurityGroup = new ec2.SecurityGroup(this, "TaskSecurityGroup", {
      vpc,
      description: "Security group for DFR-PINNs Fargate tasks",
      allowAllOutbound: true,
    });

    // Outputs
    new cdk.CfnOutput(this, "ClusterName", {
      value: this.cluster.clusterName,
      description: "ECS Cluster Name",
      exportName: `${this.stackName}-ClusterName`,
    });

    new cdk.CfnOutput(this, "ClusterArn", {
      value: this.cluster.clusterArn,
      description: "ECS Cluster ARN",
      exportName: `${this.stackName}-ClusterArn`,
    });

    new cdk.CfnOutput(this, "TaskDefinitionArn", {
      value: this.taskDefinition.taskDefinitionArn,
      description: "Task Definition ARN",
      exportName: `${this.stackName}-TaskDefinitionArn`,
    });

    // Output example run command
    new cdk.CfnOutput(this, "RunTaskCommand", {
      value: `aws ecs run-task --cluster ${this.cluster.clusterName} --task-definition ${this.taskDefinition.taskDefinitionArn} --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[${vpc.privateSubnets.map((s) => s.subnetId).join(",")}],securityGroups=[${taskSecurityGroup.securityGroupId}]}" --overrides '{"containerOverrides":[{"name":"ExperimentContainer","command":["dfr-pinns","run","--method","dfr","--problem","sine"]}]}'`,
      description: "Example command to run an experiment task",
    });
  }
}
