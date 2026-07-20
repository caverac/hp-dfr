import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { type Construct } from "constructs";

export interface GitHubOIDCStackProps extends cdk.StackProps {
  /** GitHub repository in `owner/repo` form. @example "caverac/hp-dfr" */
  readonly githubRepo: string;
  /** GitHub environments allowed to assume the role. Defaults to development only. */
  readonly allowedEnvironments?: readonly string[];
  /**
   * ARN of an existing GitHub OIDC provider to reuse. AWS allows only one
   * provider per URL per account, so when the account already has one (from
   * another project), pass its ARN to import it instead of creating a duplicate.
   */
  readonly existingProviderArn?: string;
}

/**
 * The OIDC provider and IAM role GitHub Actions assumes to deploy, so CI needs
 * no long-lived AWS credentials. The role can assume the CDK bootstrap roles,
 * which is all `cdk deploy` requires. Deploy this once, manually, with admin
 * credentials before wiring up CI.
 */
export class GitHubOIDCStack extends cdk.Stack {
  public readonly role: iam.Role;

  constructor(scope: Construct, id: string, props: GitHubOIDCStackProps) {
    super(scope, id, props);

    const { githubRepo, allowedEnvironments = ["development"], existingProviderArn } = props;

    // AWS allows only one OIDC provider per URL per account: reuse the existing
    // one when given, otherwise create it.
    const providerArn =
      existingProviderArn ??
      new iam.OpenIdConnectProvider(this, "GitHubOIDCProvider", {
        url: "https://token.actions.githubusercontent.com",
        clientIds: ["sts.amazonaws.com"],
      }).openIdConnectProviderArn;

    const subjectConditions = allowedEnvironments.map(
      (environment) => `repo:${githubRepo}:environment:${environment}`
    );

    this.role = new iam.Role(this, "GitHubActionsRole", {
      roleName: "DfrPinns-GitHubActions-Role",
      description: "Role assumed by GitHub Actions for hp-DFR deployments",
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: new iam.WebIdentityPrincipal(providerArn, {
        StringEquals: {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
        },
        StringLike: {
          "token.actions.githubusercontent.com:sub": subjectConditions,
        },
      }),
    });

    // `cdk deploy` works entirely through the bootstrap roles.
    this.role.addToPolicy(
      new iam.PolicyStatement({
        sid: "AssumeCDKBootstrapRoles",
        effect: iam.Effect.ALLOW,
        actions: ["sts:AssumeRole"],
        resources: ["arn:aws:iam::*:role/cdk-*"],
      })
    );

    new cdk.CfnOutput(this, "RoleArn", {
      value: this.role.roleArn,
      description: "IAM role ARN for GitHub Actions OIDC authentication",
    });
  }
}
