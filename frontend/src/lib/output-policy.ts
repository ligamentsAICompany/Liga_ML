import type { CloudProviderId, OutputPolicy } from '../types/agent.js';

export type OutputPolicyOption = {
  value: OutputPolicy;
  label: string;
};

type OutputPolicyProvider = CloudProviderId | 'aws-sagemaker' | string;

const OUTPUT_POLICIES: OutputPolicy[] = ['cloud-private', 'hf-hub', 'cloud-and-hf-hub'];

function cloudStorageLabel(provider: OutputPolicyProvider): string {
  switch (provider) {
    case 'gcp-vertex':
      return 'Google Cloud Storage';
    case 'aws-sagemaker':
      return 'AWS S3';
    case 'hf-jobs':
      return 'Hugging Face job artifacts';
    default:
      return 'provider-native storage';
  }
}

export function outputPolicyLabel(
  provider: OutputPolicyProvider,
  policy: OutputPolicy,
): string {
  if (provider === 'hf-jobs') {
    switch (policy) {
      case 'cloud-private':
        return 'Private Hugging Face job/model artifacts';
      case 'hf-hub':
        return 'Hugging Face Hub';
      case 'cloud-and-hf-hub':
        return 'Hugging Face Hub and job artifacts';
    }
  }

  const storageLabel = cloudStorageLabel(provider);
  switch (policy) {
    case 'cloud-private':
      return `${storageLabel} only`;
    case 'hf-hub':
      return 'Hugging Face Hub only';
    case 'cloud-and-hf-hub':
      return `Both ${storageLabel} and Hugging Face Hub`;
  }
}

export function storageDestinationLabel(
  provider: OutputPolicyProvider,
  policy: OutputPolicy,
): string {
  if (provider === 'hf-jobs') {
    return outputPolicyLabel(provider, policy);
  }

  const storageLabel = cloudStorageLabel(provider);
  switch (policy) {
    case 'cloud-private':
      return `${storageLabel} only`;
    case 'hf-hub':
      return 'Hugging Face Hub only';
    case 'cloud-and-hf-hub':
      return `${storageLabel} and Hugging Face Hub`;
  }
}

export function outputPolicyOptionsForProvider(
  provider: OutputPolicyProvider,
): OutputPolicyOption[] {
  return OUTPUT_POLICIES.map((value) => ({
    value,
    label: outputPolicyLabel(provider, value),
  }));
}

export function outputPolicyRequiresHub(policy: OutputPolicy): boolean {
  return policy === 'hf-hub' || policy === 'cloud-and-hf-hub';
}

export function outputPolicyRequiresCloudStorage(policy: OutputPolicy): boolean {
  return policy === 'cloud-private' || policy === 'cloud-and-hf-hub';
}
