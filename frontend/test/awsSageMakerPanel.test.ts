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

test('AWS final summary includes S3 model artifact and final model URL', () => {
  const output = appendAwsTrainingResultSummary(`
LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=aws-sagemaker
LIGA_AWS_TRAINING_JOB_NAME=training-job-1
LIGA_S3_MODEL_ARTIFACT=s3://bucket/prefix/output/model.tar.gz
LIGA_FINAL_MODEL_URL=https://huggingface.co/alice/aws-model
`);

  assert.match(output, /Liga Training Result/);
  assert.match(output, /s3:\/\/bucket\/prefix\/output\/model.tar.gz/);
  assert.match(output, /https:\/\/huggingface.co\/alice\/aws-model/);
});
