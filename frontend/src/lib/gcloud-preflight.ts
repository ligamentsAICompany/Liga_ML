import type { CloudProviderId, OutputPolicy, TrainingGoal } from '../types/agent.js';
import {
  outputPolicyLabel as sharedOutputPolicyLabel,
  storageDestinationLabel as sharedStorageDestinationLabel,
} from './output-policy.js';

export const DEFAULT_TRAINING_GOAL: TrainingGoal = 'agent-decide';
export const DEFAULT_OUTPUT_POLICY: OutputPolicy = 'cloud-and-hf-hub';

export const TRAINING_GOAL_OPTIONS: Array<{
  value: TrainingGoal;
  label: string;
}> = [
  { value: 'smoke-test', label: 'Quick smoke test' },
  { value: 'production', label: 'Production-ready fine-tuning' },
  { value: 'agent-decide', label: 'Let the agent decide' },
];

export const OUTPUT_POLICY_OPTIONS: Array<{
  value: OutputPolicy;
  label: string;
}> = [
  { value: 'cloud-private', label: sharedOutputPolicyLabel('gcp-vertex', 'cloud-private') },
  { value: 'hf-hub', label: sharedOutputPolicyLabel('gcp-vertex', 'hf-hub') },
  {
    value: 'cloud-and-hf-hub',
    label: sharedOutputPolicyLabel('gcp-vertex', 'cloud-and-hf-hub'),
  },
];

export function trainingGoalLabel(value: TrainingGoal | undefined): string {
  return TRAINING_GOAL_OPTIONS.find((option) => option.value === value)?.label
    ?? trainingGoalLabel(DEFAULT_TRAINING_GOAL);
}

export function outputPolicyLabel(
  value: OutputPolicy | undefined,
  provider: CloudProviderId = 'gcp-vertex',
): string {
  const policy = value ?? DEFAULT_OUTPUT_POLICY;
  return sharedOutputPolicyLabel(provider, policy);
}

export function storageDestinationLabel(
  value: OutputPolicy | undefined,
  provider: CloudProviderId = 'gcp-vertex',
): string {
  return sharedStorageDestinationLabel(provider, value ?? DEFAULT_OUTPUT_POLICY);
}

export function buildGcloudChatRequestMetadata({
  cloudProvider,
  trainingGoal,
  outputPolicy,
}: {
  cloudProvider: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
}): {
  cloud_provider: CloudProviderId;
  training_goal?: TrainingGoal;
  output_policy?: OutputPolicy;
} {
  return {
    cloud_provider: cloudProvider,
    training_goal: trainingGoal ?? DEFAULT_TRAINING_GOAL,
    output_policy: outputPolicy ?? DEFAULT_OUTPUT_POLICY,
  };
}
