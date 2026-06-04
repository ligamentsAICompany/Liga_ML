import assert from 'node:assert/strict';
import { test } from 'node:test';

import { parseLigaTrainingResult } from '../src/utils/trainingResult.js';

test('parses all Liga final result markers from repeated logs', () => {
  const result = parseLigaTrainingResult(`
noise before
LIGA_TRAINING_STATUS=running
LIGA_PROVIDER=gcp-vertex
LIGA_FINAL_MODEL_URL=https://huggingface.co/alice/model-v1.
LIGA_HUB_MODEL_ID=alice/model-v1
LIGA_GCS_OUTPUT_DIR=gs://liga-training/vertex-outputs/run-1
LIGA_EVAL_RESULT_JSON={"eval_loss":0.42,"samples":3}
LIGA_RESULT_FILE=liga_training_result.json
LIGA_DATASET_SOURCE=gcs_jsonl
LIGA_STAGED_TRAIN_URI=gs://liga-training/vertex-inputs/run-1/train.jsonl
LIGA_TRAIN_ROWS=10
LIGA_EVAL_ROWS=2
more logs
LIGA_TRAINING_STATUS=succeeded
`);

  assert.deepEqual(result, {
    status: 'succeeded',
    provider: 'gcp-vertex',
    finalModelUrl: 'https://huggingface.co/alice/model-v1',
    hubModelId: 'alice/model-v1',
    gcsOutputDir: 'gs://liga-training/vertex-outputs/run-1',
    datasetSource: 'gcs_jsonl',
    stagedTrainUri: 'gs://liga-training/vertex-inputs/run-1/train.jsonl',
    trainRows: '10',
    evalRows: '2',
    evalResult: { eval_loss: 0.42, samples: 3 },
    resultFile: 'liga_training_result.json',
  });
});

test('tolerates malformed eval JSON without throwing', () => {
  const result = parseLigaTrainingResult(`
LIGA_PROVIDER=gcp-vertex
LIGA_EVAL_RESULT_JSON={"eval_loss":
LIGA_RESULT_FILE=liga_training_result.json
`);

  assert.equal(result?.provider, 'gcp-vertex');
  assert.equal(result?.evalResult, null);
  assert.equal(result?.resultFile, 'liga_training_result.json');
});

test('returns null when no final result markers exist', () => {
  assert.equal(
    parseLigaTrainingResult('Training complete at https://huggingface.co/alice/not-a-marker'),
    null,
  );
});

test('parses missing optional markers and empty eval object generically', () => {
  const result = parseLigaTrainingResult(`
LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=gcp-vertex
LIGA_EVAL_RESULT_JSON={}
`);

  assert.deepEqual(result, {
    status: 'succeeded',
    provider: 'gcp-vertex',
    evalResult: {},
  });
});

test('strips safe trailing URL punctuation from marked final model URL', () => {
  const result = parseLigaTrainingResult(
    'LIGA_FINAL_MODEL_URL=https://huggingface.co/alice/model),',
  );

  assert.equal(result?.finalModelUrl, 'https://huggingface.co/alice/model');
});

test('parses AWS SageMaker final result markers', () => {
  const result = parseLigaTrainingResult(`
LIGA_TRAINING_STATUS=succeeded
LIGA_PROVIDER=aws-sagemaker
LIGA_AWS_TRAINING_JOB_NAME=training-job-1
LIGA_AWS_REGION=us-east-1
LIGA_S3_MODEL_ARTIFACT=s3://bucket/prefix/output/model.tar.gz
LIGA_S3_OUTPUT_DIR=s3://bucket/prefix/output/
LIGA_CLOUDWATCH_LOGS_URL=https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/foo,
LIGA_FINAL_MODEL_URL=https://huggingface.co/alice/aws-model.
LIGA_EVAL_RESULT_JSON={"eval_loss":0.25}
LIGA_RESULT_FILE=liga_training_result.json
`);

  assert.deepEqual(result, {
    status: 'succeeded',
    provider: 'aws-sagemaker',
    awsTrainingJobName: 'training-job-1',
    awsRegion: 'us-east-1',
    s3ModelArtifact: 's3://bucket/prefix/output/model.tar.gz',
    s3OutputDir: 's3://bucket/prefix/output/',
    cloudWatchLogsUrl: 'https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/foo',
    finalModelUrl: 'https://huggingface.co/alice/aws-model',
    evalResult: { eval_loss: 0.25 },
    resultFile: 'liga_training_result.json',
  });
});

test('uses latest repeated AWS marker values', () => {
  const result = parseLigaTrainingResult(`
LIGA_PROVIDER=aws-sagemaker
LIGA_AWS_TRAINING_JOB_NAME=old-job
LIGA_S3_MODEL_ARTIFACT=s3://bucket/old/model.tar.gz
LIGA_AWS_TRAINING_JOB_NAME=new-job
LIGA_S3_MODEL_ARTIFACT=s3://bucket/new/model.tar.gz
`);

  assert.equal(result?.awsTrainingJobName, 'new-job');
  assert.equal(result?.s3ModelArtifact, 's3://bucket/new/model.tar.gz');
});

test('parses AWS result with missing optional markers', () => {
  const result = parseLigaTrainingResult(`
LIGA_PROVIDER=aws-sagemaker
LIGA_AWS_TRAINING_JOB_NAME=training-job-1
`);

  assert.deepEqual(result, {
    provider: 'aws-sagemaker',
    awsTrainingJobName: 'training-job-1',
  });
});
