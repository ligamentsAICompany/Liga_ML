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
more logs
LIGA_TRAINING_STATUS=succeeded
`);

  assert.deepEqual(result, {
    status: 'succeeded',
    provider: 'gcp-vertex',
    finalModelUrl: 'https://huggingface.co/alice/model-v1',
    hubModelId: 'alice/model-v1',
    gcsOutputDir: 'gs://liga-training/vertex-outputs/run-1',
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
