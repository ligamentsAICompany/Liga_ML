import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  outputPolicyLabel,
  outputPolicyOptionsForProvider,
  outputPolicyRequiresCloudStorage,
  outputPolicyRequiresHub,
  storageDestinationLabel,
} from '../src/lib/output-policy.js';
import type { CloudProviderId, OutputPolicy } from '../src/types/agent.js';

test('formats provider-specific output policy labels', () => {
  assert.equal(outputPolicyLabel('gcp-vertex', 'cloud-private'), 'Google Cloud Storage only');
  assert.equal(outputPolicyLabel('gcp-vertex', 'hf-hub'), 'Hugging Face Hub only');
  assert.equal(
    outputPolicyLabel('gcp-vertex', 'cloud-and-hf-hub'),
    'Both Google Cloud Storage and Hugging Face Hub',
  );

  assert.equal(outputPolicyLabel('aws-sagemaker' as CloudProviderId, 'cloud-private'), 'AWS S3 only');
  assert.equal(
    outputPolicyLabel('aws-sagemaker' as CloudProviderId, 'cloud-and-hf-hub'),
    'Both AWS S3 and Hugging Face Hub',
  );

  assert.equal(
    outputPolicyLabel('hf-jobs', 'cloud-private'),
    'Private Hugging Face job/model artifacts',
  );
  assert.equal(outputPolicyLabel('hf-jobs', 'hf-hub'), 'Hugging Face Hub');
  assert.equal(
    outputPolicyLabel('hf-jobs', 'cloud-and-hf-hub'),
    'Hugging Face Hub and job artifacts',
  );
});

test('formats storage destination labels from provider and policy', () => {
  assert.equal(storageDestinationLabel('gcp-vertex', 'cloud-private'), 'Google Cloud Storage only');
  assert.equal(storageDestinationLabel('gcp-vertex', 'hf-hub'), 'Hugging Face Hub only');
  assert.equal(
    storageDestinationLabel('gcp-vertex', 'cloud-and-hf-hub'),
    'Google Cloud Storage and Hugging Face Hub',
  );
  assert.equal(storageDestinationLabel('aws-sagemaker' as CloudProviderId, 'cloud-private'), 'AWS S3 only');
  assert.equal(
    storageDestinationLabel('hf-jobs', 'cloud-and-hf-hub'),
    'Hugging Face Hub and job artifacts',
  );
});

test('keeps the three canonical options for every provider', () => {
  const expected: OutputPolicy[] = ['cloud-private', 'hf-hub', 'cloud-and-hf-hub'];

  for (const provider of ['gcp-vertex', 'hf-jobs', 'aws-sagemaker'] as CloudProviderId[]) {
    assert.deepEqual(
      outputPolicyOptionsForProvider(provider).map((option) => option.value),
      expected,
    );
  }
});

test('detects whether output policy requires Hub or provider storage', () => {
  assert.equal(outputPolicyRequiresHub('cloud-private'), false);
  assert.equal(outputPolicyRequiresHub('hf-hub'), true);
  assert.equal(outputPolicyRequiresHub('cloud-and-hf-hub'), true);

  assert.equal(outputPolicyRequiresCloudStorage('cloud-private'), true);
  assert.equal(outputPolicyRequiresCloudStorage('hf-hub'), false);
  assert.equal(outputPolicyRequiresCloudStorage('cloud-and-hf-hub'), true);
});

test('vertex panel source masks secret-like parameters before storing panel data', () => {
  const source = readFileSync('src/lib/vertex-job-panel.ts', 'utf8');

  assert.match(source, /SECRET_KEY_PATTERN/);
  assert.match(source, /maskSensitiveParameters/);
  assert.doesNotMatch(source, /HF_TOKEN.*parameters: args/);
});
