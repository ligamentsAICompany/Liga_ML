import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildVertexStateMarkdown,
  createVertexRunPanel,
} from '../src/lib/vertex-job-panel.js';

test('renders failed Vertex state with root error even when logs are unavailable', () => {
  const markdown = buildVertexStateMarkdown({
    state: 'failed',
    jobName: 'projects/test-project/locations/us-central1/customJobs/123',
    jobUrl: 'https://console.cloud.google.com/vertex-ai/training/custom-jobs/locations/us-central1/customJobs/123?project=test-project',
    outputDir: 'gs://liga-training/vertex-outputs/job',
    failureReason: "DatasetNotFoundError: Dataset 'owner/private' cannot be accessed.",
    logsUnavailable: true,
  });

  assert.match(markdown, /failed/);
  assert.match(markdown, /DatasetNotFoundError/);
  assert.match(markdown, /Logs are not available yet, but Vertex already reported failure/);
  assert.match(markdown, /console.cloud.google.com/);
});

test('renders staged uploaded dataset URI in Vertex SFT run panel', () => {
  const panel = createVertexRunPanel({
    operation: 'run',
    template: 'sft',
    dataset_name: 'owner/session-datasets',
    dataset_config: 'upload_abc',
    dataset_source: 'uploaded-gcs',
    staged_train_uri: 'gs://liga-training/vertex-inputs/job/train.jsonl',
    model_name: 'Qwen/Qwen2.5-0.5B-Instruct',
    output_policy: 'cloud-private',
  });

  assert.ok(panel);
  assert.match(panel.data.output?.content ?? '', /uploaded-gcs/);
  assert.match(panel.data.output?.content ?? '', /gs:\/\/liga-training\/vertex-inputs\/job\/train\.jsonl/);
});
