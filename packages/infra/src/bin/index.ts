#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";

import { parseEnv } from "@infra/env";
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
