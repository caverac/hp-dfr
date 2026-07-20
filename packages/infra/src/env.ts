/** Typed parsing of the environment variables the CDK app is driven by. */

import { z } from "zod";

export type DeploymentEnvironment = "development" | "production";

const EnvSchema = z.object({
  ENVIRONMENT: z.enum(["development", "production"]),
  AWS_ACCOUNT: z.string().regex(/^\d{12}$/, "AWS_ACCOUNT must be a 12-digit account id"),
  AWS_REGION: z.string().min(1).default("us-east-1"),
  // For the GitHub Actions OIDC trust policy. Defaults to this repo so a manual
  // deploy needs only ENVIRONMENT and AWS_ACCOUNT; CI passes github.repository.
  GITHUB_REPOSITORY: z.string().min(1).default("caverac/hp-dfr"),
  // AWS allows one OIDC provider per URL per account; reuse an existing one.
  REUSE_GITHUB_OIDC_PROVIDER: z.enum(["true", "false"]).default("false"),
});

export interface CdkEnv {
  readonly environment: DeploymentEnvironment;
  /** Target AWS account id (12 digits). Each environment is a separate account. */
  readonly account: string;
  /** Target AWS region. */
  readonly region: string;
  /** GitHub repository in `owner/repo` form, for the OIDC trust policy. */
  readonly githubRepo: string;
  /** Reuse the account's existing GitHub OIDC provider instead of creating one. */
  readonly reuseGithubOidcProvider: boolean;
}

/** Parse and validate the CDK environment, throwing on anything missing or malformed. */
export function parseEnv(source: NodeJS.ProcessEnv): CdkEnv {
  const env = EnvSchema.parse(source);
  return {
    environment: env.ENVIRONMENT,
    account: env.AWS_ACCOUNT,
    region: env.AWS_REGION,
    githubRepo: env.GITHUB_REPOSITORY,
    reuseGithubOidcProvider: env.REUSE_GITHUB_OIDC_PROVIDER === "true",
  };
}
