#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";

import { parseEnv } from "@infra/env";
import { GitHubOIDCStack } from "@infra/lib/github-oidc.stack";
import { SweepStack } from "@infra/lib/sweep.stack";

const env = parseEnv(process.env);

const app = new cdk.App();

// Each environment is a separate AWS account, so the account is the
// discriminator and the stack needs no environment suffix.
new SweepStack(app, "DfrPinnsSweep", {
  environment: env.environment,
  env: { account: env.account, region: env.region },
  description: `hp-DFR sweeps (${env.environment})`,
});

// The role GitHub Actions assumes to deploy. Deployed once, manually, before CI
// can authenticate; development only (see .github/workflows/deploy-development.yml).
new GitHubOIDCStack(app, "DfrPinnsOidc", {
  githubRepo: env.githubRepo,
  existingProviderArn: env.reuseGithubOidcProvider
    ? `arn:aws:iam::${env.account}:oidc-provider/token.actions.githubusercontent.com`
    : undefined,
  env: { account: env.account, region: env.region },
  description: "GitHub Actions OIDC role for hp-DFR deployments",
});
