import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { EnvironmentConfig } from "../../config/environments";

export interface StorageStackProps extends cdk.StackProps {
  config: EnvironmentConfig;
}

/**
 * Storage Stack for DFR-PINNs experiment results.
 *
 * Creates an S3 bucket for storing experiment outputs, logs,
 * and trained model checkpoints.
 */
export class StorageStack extends cdk.Stack {
  /** S3 bucket for experiment results */
  public readonly resultsBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    const { config } = props;

    // Create S3 bucket for results
    this.resultsBucket = new s3.Bucket(this, "ResultsBucket", {
      bucketName: `dfr-pinns-results-${config.name}-${cdk.Aws.ACCOUNT_ID}`,
      removalPolicy:
        config.name === "prod"
          ? cdk.RemovalPolicy.RETAIN
          : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: config.name !== "prod",

      // Security settings
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,

      // Versioning for experiment tracking
      versioned: true,

      // Lifecycle rules
      lifecycleRules: [
        {
          id: "ExpireOldResults",
          enabled: true,
          expiration: cdk.Duration.days(config.storage.retentionDays),
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
        {
          id: "TransitionToIA",
          enabled: true,
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(30),
            },
          ],
        },
      ],

      // CORS for potential web access
      cors: [
        {
          allowedMethods: [s3.HttpMethods.GET, s3.HttpMethods.HEAD],
          allowedOrigins: ["*"],
          allowedHeaders: ["*"],
          maxAge: 3000,
        },
      ],
    });

    // Outputs
    new cdk.CfnOutput(this, "ResultsBucketName", {
      value: this.resultsBucket.bucketName,
      description: "Results S3 Bucket Name",
      exportName: `${this.stackName}-ResultsBucketName`,
    });

    new cdk.CfnOutput(this, "ResultsBucketArn", {
      value: this.resultsBucket.bucketArn,
      description: "Results S3 Bucket ARN",
      exportName: `${this.stackName}-ResultsBucketArn`,
    });
  }
}
