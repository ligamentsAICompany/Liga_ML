import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildEvaluationMarkdown,
  evaluationStatusLabel,
  type PostTrainingEvaluation,
} from '../src/lib/post-training-evaluation.js';
import { parseLigaTrainingResult } from '../src/utils/trainingResult.js';

const completedEvaluation: PostTrainingEvaluation = {
  evaluation_id: 'eval_s1_r1',
  session_id: 's1',
  run_id: 'r1',
  provider: 'aws-sagemaker',
  job_id: 'train-job',
  model_ref: 'owner/model',
  artifact_ref: 's3://bucket/output/model.tar.gz',
  dataset_ref: 'owner/hardware-dataset',
  status: 'succeeded',
  created_at: '2026-06-07T00:00:00Z',
  started_at: '2026-06-07T00:00:01Z',
  completed_at: '2026-06-07T00:00:02Z',
  evaluation_type: 'static_result_review',
  domain: 'hardware',
  task_type: 'support',
  test_prompts: ['GPU overheating after ten minutes of gaming', 'Unsafe PSU repair request'],
  results: { metric_summary: { eval_loss: 0.3 } },
  scores: {
    overall_score: 0.77,
    task_relevance_score: 0.8,
    safety_score: 0.7,
    privacy_score: 0.9,
    metric_quality_score: 0.7,
    confidence: 0.65,
  },
  safety_findings: [{ severity: 'warning', message: 'Avoid unsafe PSU repair instructions.' }],
  privacy_findings: [{ severity: 'info', message: 'No secret-like metric values were retained.' }],
  quality_summary: 'Static metrics look usable for a demo.',
  failure_summary: '',
  recommendation: 'Use for controlled demo with human review.',
  report_markdown: '## Post-Training Evaluation\nSafe static report.',
  artifact_paths: ['s3://bucket/output/model.tar.gz'],
  metadata: { mode: 'static' },
};

test('evaluation status labels include empty and terminal states', () => {
  assert.equal(evaluationStatusLabel(undefined), 'Not evaluated');
  assert.equal(evaluationStatusLabel('planned'), 'Planned');
  assert.equal(evaluationStatusLabel('succeeded'), 'Complete');
  assert.equal(evaluationStatusLabel('unavailable'), 'Unavailable');
});

test('completed evaluation markdown renders scores findings prompts and artifact links', () => {
  const markdown = buildEvaluationMarkdown(completedEvaluation);

  assert.match(markdown, /Post-Training Evaluation/);
  assert.match(markdown, /Overall.*77%/);
  assert.match(markdown, /Safety.*70%/);
  assert.match(markdown, /GPU overheating/);
  assert.match(markdown, /Unsafe PSU repair/);
  assert.match(markdown, /Avoid unsafe PSU repair/);
  assert.match(markdown, /s3:\/\/bucket\/output\/model\.tar\.gz/);
  assert.match(markdown, /Use for controlled demo/);
});

test('evaluation markdown redacts secret-like values', () => {
  const markdown = buildEvaluationMarkdown({
    ...completedEvaluation,
    report_markdown: 'token=hf_abcdefghijklmnopqrstuvwxyz1234567890',
    metadata: { authorization: 'Bearer abcdefghijklmnopqrstuvwxyz123456' },
  });

  assert.doesNotMatch(markdown, /hf_abcdefghijklmnopqrstuvwxyz/);
  assert.doesNotMatch(markdown, /Bearer abcdef/);
  assert.match(markdown, /\[REDACTED\]/);
});

test('training result parses embedded post-training evaluation JSON', () => {
  const result = parseLigaTrainingResult(`
LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=aws-sagemaker
LIGA_POST_TRAINING_EVALUATION_JSON={"status":"succeeded","scores":{"overall_score":0.82},"recommendation":"Demo ready"}
`);

  assert.equal(result?.postTrainingEvaluation?.status, 'succeeded');
  assert.deepEqual(result?.postTrainingEvaluation?.scores, { overall_score: 0.82 });
});
