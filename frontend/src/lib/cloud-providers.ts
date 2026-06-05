import type { CloudProviderId } from '../types/agent.js';

export const CLOUD_PROVIDER_OPTIONS: Array<{
  id: CloudProviderId;
  name: string;
  description: string;
}> = [
  {
    id: 'hf-jobs',
    name: 'Hugging Face Jobs',
    description: 'Run training on Hugging Face infrastructure',
  },
  {
    id: 'gcp-vertex',
    name: 'Google Cloud Vertex AI',
    description: 'Run training with the gcp_vertex_jobs backend',
  },
  {
    id: 'aws-sagemaker',
    name: 'AWS SageMaker AI',
    description: 'Run training with the aws_sagemaker_jobs backend',
  },
];

const CLOUD_PROVIDER_IDS = new Set<CloudProviderId>(
  CLOUD_PROVIDER_OPTIONS.map((provider) => provider.id),
);

export function isCloudProviderId(value: unknown): value is CloudProviderId {
  return typeof value === 'string' && CLOUD_PROVIDER_IDS.has(value as CloudProviderId);
}
