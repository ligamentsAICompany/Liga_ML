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
