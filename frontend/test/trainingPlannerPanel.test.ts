import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { createTrainingPlannerPanel } from '../src/lib/training-planner-panel.js';

const toolCallGroupSource = readFileSync('src/components/Chat/ToolCallGroup.tsx', 'utf8');

test('training planner panel renders recommended and staged models', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'gcp-vertex',
    trainingGoal: 'production',
    recommendedModel: 'meta-llama/Llama-3.2-3B-Instruct',
    smokeTestModel: 'Qwen/Qwen2.5-0.5B-Instruct',
    productionModel: 'meta-llama/Llama-3.2-3B-Instruct',
  });

  assert.equal(panel.title, 'Training Planner Recommendation');
  assert.match(panel.markdown, /Recommended model: meta-llama\/Llama-3\.2-3B-Instruct/);
  assert.match(panel.markdown, /Smoke-test model: Qwen\/Qwen2\.5-0\.5B-Instruct/);
  assert.match(panel.markdown, /Production model: meta-llama\/Llama-3\.2-3B-Instruct/);
});

test('training planner panel renders provider hardware and output policy labels', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'aws-sagemaker',
    recommendedHardware: {
      instance_type: 'ml.g5.2xlarge',
      instance_count: 1,
    },
    outputPolicy: 'cloud-private',
  });

  assert.match(panel.markdown, /Hardware: instance type: ml\.g5\.2xlarge, instance count: 1/);
  assert.match(panel.markdown, /Output policy: AWS S3 only/);
});

test('training planner panel renders privacy warnings and risks', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'hf-jobs',
    domain: 'finance',
    outputPolicy: 'cloud-private',
    privacyWarnings: ['Sensitive data detected; prefer private storage.'],
    risks: ['No training dataset summary is available.'],
  });

  assert.match(panel.markdown, /Sensitive data detected; prefer private storage\./);
  assert.match(panel.markdown, /No training dataset summary is available\./);
  assert.match(panel.markdown, /User approval is required before any billable cloud job\./);
});

test('training planner panel renders phase 7 recommendation sections', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'aws-sagemaker',
    trainingGoal: 'production',
    recommendedModel: 'Qwen/Qwen2.5-1.5B-Instruct',
    recommendedHardware: {
      instance_type: 'ml.g4dn.xlarge',
      estimated_hourly_cost_usd: 0.9,
    },
    outputPolicy: 'cloud-private',
    recommendation: {
      selected_model: {
        model_id: 'Qwen/Qwen2.5-1.5B-Instruct',
        family: 'Qwen',
        parameter_count_b: 1.5,
        license: 'apache-2.0',
        gated: false,
      },
      selected_provider: {
        provider_id: 'aws-sagemaker',
        display_name: 'AWS SageMaker AI',
      },
      selected_hardware: {
        hardware_id: 'aws-sagemaker:ml.g4dn.xlarge',
        display_name: 'ml.g4dn.xlarge',
        gpu_memory_gb: 16,
      },
      estimated_cost_usd: 1.8,
      budget_cap_usd: 10,
      confidence: 0.78,
      warnings: [{ message: 'ml.g5.xlarge quota is 0; using ml.g4dn.xlarge fallback.' }],
      fallbacks: [
        {
          blocked_option: 'aws-sagemaker:ml.g5.xlarge',
          fallback_option: 'aws-sagemaker:ml.g4dn.xlarge',
          reason: 'quota unavailable',
        },
      ],
      production_alternative: {
        model_id: 'Qwen/Qwen2.5-3B-Instruct',
        hardware_id: 'aws-sagemaker:ml.g5.2xlarge',
      },
      recommended_evaluation_profile: 'safety_privacy_review',
    },
  });

  assert.match(panel.markdown, /### Primary recommendation/);
  assert.match(panel.markdown, /Model: Qwen\/Qwen2\.5-1\.5B-Instruct/);
  assert.match(panel.markdown, /License: apache-2\.0/);
  assert.match(panel.markdown, /Provider: AWS SageMaker AI/);
  assert.match(panel.markdown, /Hardware: ml\.g4dn\.xlarge/);
  assert.match(panel.markdown, /Estimated cost: \$1\.80/);
  assert.match(panel.markdown, /Budget cap: \$10\.00/);
  assert.match(panel.markdown, /quota is 0/);
  assert.match(panel.markdown, /Fallback: aws-sagemaker:ml\.g5\.xlarge -> aws-sagemaker:ml\.g4dn\.xlarge/);
  assert.match(panel.markdown, /Production alternative: Qwen\/Qwen2\.5-3B-Instruct on aws-sagemaker:ml\.g5\.2xlarge/);
});

test('training planner panel renders Vertex provider hardware output policy and cost', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'gcp-vertex',
    trainingGoal: 'smoke-test',
    recommendedModel: 'Qwen/Qwen2.5-0.5B-Instruct',
    recommendedHardware: {
      machine_type: 'n1-standard-8',
      accelerator_type: 'NVIDIA_TESLA_T4',
      accelerator_count: 1,
    },
    outputPolicy: 'cloud-private',
    recommendation: {
      selected_model: {
        model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
        family: 'Qwen',
        parameter_count_b: 0.5,
        license: 'apache-2.0',
        access: 'open',
      },
      selected_provider: {
        provider_id: 'gcp-vertex',
        display_name: 'Google Cloud Vertex AI',
      },
      selected_hardware: {
        hardware_id: 'gcp-vertex:n1-standard-8-t4',
        display_name: 'n1-standard-8 + T4',
        gpu_memory_gb: 16,
      },
      estimated_cost_usd: 1.1,
      warnings: [{ message: 'GCloud readiness is unknown; verify configuration.' }],
      fallbacks: [],
    },
  });

  assert.match(panel.markdown, /Provider: Google Cloud Vertex AI/);
  assert.match(panel.markdown, /Provider: gcp-vertex/);
  assert.match(panel.markdown, /Hardware: n1-standard-8 \+ T4/);
  assert.match(panel.markdown, /Hardware id: gcp-vertex:n1-standard-8-t4/);
  assert.match(panel.markdown, /Output policy: Google Cloud Storage only/);
  assert.match(panel.markdown, /Estimated cost: \$1\.10/);
});

test('training planner panel redacts secret-looking recommendation values', () => {
  const panel = createTrainingPlannerPanel({
    provider: 'hf-jobs',
    recommendation: {
      warnings: [{ message: 'Do not show sk-secret123456 in UI.' }],
    },
  });

  assert.doesNotMatch(panel.markdown, /sk-secret123456/);
  assert.match(panel.markdown, /\[REDACTED\]/);
});

test('training planner panel handles missing optional fields gracefully', () => {
  const panel = createTrainingPlannerPanel({});

  assert.match(panel.markdown, /Recommended model: Not specified yet/);
  assert.match(panel.markdown, /Planning only/);
  assert.doesNotMatch(panel.markdown, /undefined|null/);
});

test('training planner tool displays a readable label', () => {
  assert.match(toolCallGroupSource, /training_planner/);
  assert.match(toolCallGroupSource, /Training Planner/);
});

test('tool call group persists parsed HF job status outside render', () => {
  assert.match(toolCallGroupSource, /Persist parsed HF job status outside render/);
  assert.doesNotMatch(
    toolCallGroupSource,
    /setJobStatus\(tool\.toolCallId,\s*jobMetaFromOutput\.jobStatus\);/,
  );
});
