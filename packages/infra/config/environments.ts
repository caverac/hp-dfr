/**
 * Environment configuration for DFR-PINNs infrastructure.
 */

export interface EnvironmentConfig {
  /** Environment name (dev, staging, prod) */
  name: string;
  /** AWS account ID */
  account: string;
  /** AWS region */
  region: string;
  /** Fargate task configuration */
  fargate: {
    cpu: number;
    memoryMiB: number;
    desiredCount: number;
  };
  /** S3 results bucket configuration */
  storage: {
    retentionDays: number;
  };
}

/**
 * Default development configuration.
 * Override account and region with your own values.
 */
export const devConfig: EnvironmentConfig = {
  name: "dev",
  account: process.env.CDK_DEFAULT_ACCOUNT || "",
  region: process.env.CDK_DEFAULT_REGION || "us-east-1",
  fargate: {
    cpu: 1024, // 1 vCPU
    memoryMiB: 2048, // 2 GB
    desiredCount: 0, // Start with no tasks, scale on demand
  },
  storage: {
    retentionDays: 30,
  },
};

/**
 * Production configuration with larger resources.
 */
export const prodConfig: EnvironmentConfig = {
  name: "prod",
  account: process.env.CDK_DEFAULT_ACCOUNT || "",
  region: process.env.CDK_DEFAULT_REGION || "us-east-1",
  fargate: {
    cpu: 4096, // 4 vCPUs
    memoryMiB: 8192, // 8 GB
    desiredCount: 0,
  },
  storage: {
    retentionDays: 365,
  },
};

/**
 * Get configuration by environment name.
 */
export function getConfig(env: string = "dev"): EnvironmentConfig {
  switch (env) {
    case "prod":
      return prodConfig;
    case "dev":
    default:
      return devConfig;
  }
}
