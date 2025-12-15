import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

/**
 * ECR Stack for DFR-PINNs container images.
 *
 * Creates an ECR repository for storing experiment container images.
 */
export class EcrStack extends cdk.Stack {
  /** The ECR repository for experiment images */
  public readonly repository: ecr.Repository;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create ECR repository
    this.repository = new ecr.Repository(this, "ExperimentRepo", {
      repositoryName: "dfr-pinns-experiments",
      removalPolicy: cdk.RemovalPolicy.DESTROY, // Destroy images on stack deletion
      imageScanOnPush: true, // Scan for vulnerabilities
      lifecycleRules: [
        {
          description: "Keep only 10 untagged images",
          maxImageCount: 10,
          tagStatus: ecr.TagStatus.UNTAGGED,
        },
        {
          description: "Expire old tagged images after 10 days",
          maxImageAge: cdk.Duration.days(10),
          tagStatus: ecr.TagStatus.TAGGED,
          tagPrefixList: ["dev-", "test-"],
        },
      ],
    });

    // Outputs
    new cdk.CfnOutput(this, "RepositoryUri", {
      value: this.repository.repositoryUri,
      description: "ECR Repository URI",
      exportName: `${this.stackName}-RepositoryUri`,
    });

    new cdk.CfnOutput(this, "RepositoryArn", {
      value: this.repository.repositoryArn,
      description: "ECR Repository ARN",
      exportName: `${this.stackName}-RepositoryArn`,
    });
  }
}
