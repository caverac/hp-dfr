import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

/**
 * VPC Stack for DFR-PINNs infrastructure.
 *
 * Creates a VPC with public and private subnets for running
 * Fargate tasks with internet access for pulling container images.
 */
export class VpcStack extends cdk.Stack {
  /** The VPC for experiment infrastructure */
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create VPC with public and private subnets
    this.vpc = new ec2.Vpc(this, "ExperimentVpc", {
      maxAzs: 2, // Use 2 availability zones for redundancy
      natGateways: 1, // Single NAT gateway to reduce costs
      subnetConfiguration: [
        {
          name: "Public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: "Private",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
      ],
    });

    // VPC Flow Logs for debugging (optional, can be removed for cost savings)
    this.vpc.addFlowLog("FlowLog", {
      destination: ec2.FlowLogDestination.toCloudWatchLogs(),
      trafficType: ec2.FlowLogTrafficType.REJECT,
    });

    // Outputs
    new cdk.CfnOutput(this, "VpcId", {
      value: this.vpc.vpcId,
      description: "VPC ID",
      exportName: `${this.stackName}-VpcId`,
    });
  }
}
