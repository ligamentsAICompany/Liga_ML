import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  appendAwsTrainingResultSummary,
  buildAwsStateMarkdown,
  createAwsSageMakerRunPanel,
} from '../src/lib/aws-sagemaker-panel.js';

test('AWS run panel includes dataset model and output policy', () => {
  const panel = createAwsSageMakerRunPanel({
    operation: 'run',
    template: 'sft',
    dataset_name: 'owner/dataset',
    dataset_config: 'default',
    dataset_split: 'train',
    model_name: 'Qwen/Qwen2.5-0.5B-Instruct',
    output_model_id: 'owner/aws-output',
    output_policy: 'cloud-and-hf-hub',
    instance_type: 'ml.g5.xlarge',
    instance_count: 1,
    max_run_seconds: 3600,
    s3_bucket: 'training-bucket',
    s3_prefix: 'liga-ml',
  });

  assert.equal(panel?.data.title, 'AWS SageMaker SFT Training');
  assert.match(panel?.data.output?.content || '', /owner\/dataset/);
  assert.match(panel?.data.output?.content || '', /Qwen\/Qwen2.5-0.5B-Instruct/);
  assert.match(panel?.data.output?.content || '', /cloud-and-hf-hub/);
  assert.match(panel?.data.output?.content || '', /training-bucket/);
});

test('AWS running state markdown includes job S3 and CloudWatch links', () => {
  const markdown = buildAwsStateMarkdown({
    state: 'running',
    jobName: 'training-job-1',
    jobUrl: 'https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/training-job-1',
    region: 'us-east-1',
    s3TrainUri: 's3://bucket/prefix/input/train.jsonl',
    s3OutputUri: 's3://bucket/prefix/output/',
    s3ModelArtifact: 's3://bucket/prefix/output/model.tar.gz',
    cloudWatchLogsUrl: 'https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/foo',
  });

  assert.match(markdown, /training-job-1/);
  assert.match(markdown, /s3:\/\/bucket\/prefix\/output\/model.tar.gz/);
  assert.match(markdown, /CloudWatch logs/);
  assert.match(markdown, /SageMaker console/);
});

test('AWS final summary keeps S3-only outputs visible without HF URL', () => {
  const output = appendAwsTrainingResultSummary(`
LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=aws-sagemaker
LIGA_AWS_TRAINING_JOB_NAME=training-job-1
LIGA_AWS_REGION=us-east-1
LIGA_S3_MODEL_ARTIFACT=s3://bucket/prefix/output/model.tar.gz
LIGA_S3_OUTPUT_DIR=s3://bucket/prefix/output/
LIGA_CLOUDWATCH_LOGS_URL=https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/foo
LIGA_OUTPUT_POLICY=aws-private
LIGA_EVAL_RESULT_JSON={"eval_loss":0.25,"eval_samples_per_second":2}
LIGA_RESULT_FILE=liga_training_result.json
`);

  assert.match(output, /Liga Training Result/);
  assert.match(output, /succeeded/);
  assert.match(output, /aws-sagemaker/);
  assert.match(output, /s3:\/\/bucket\/prefix\/output\/model.tar.gz/);
  assert.match(output, /s3:\/\/bucket\/prefix\/output\//);
  assert.match(output, /liga_training_result\.json/);
  assert.match(output, /eval_loss/);
  assert.match(output, /aws-private/);
  assert.doesNotMatch(output, /huggingface\.co/);
});

test('AWS final summary explains missing result JSON while preserving success', () => {
  const output = appendAwsTrainingResultSummary(`
AWS training completed.

**TrainingJobStatus:** Completed
**S3ModelArtifacts:** s3://bucket/prefix/output/model.tar.gz
**Result file:** result JSON was not found separately; inspect model artifact if needed.

LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=aws-sagemaker
LIGA_S3_MODEL_ARTIFACT=s3://bucket/prefix/output/model.tar.gz
`);

  assert.match(output, /succeeded/);
  assert.match(output, /s3:\/\/bucket\/prefix\/output\/model.tar.gz/);
  assert.match(output, /result JSON was not found separately/);
});
