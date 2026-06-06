import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import {
  buildLlmErrorRecord,
  clearLlmErrorOnModelChange,
  clearLlmErrorOnNewPrompt,
  clearLlmErrorOnSuccessfulRequest,
  isModelUnavailable,
  markUnavailableModelForError,
  modelSpecificErrorCopy,
  shouldReportLlmHealthErrorForActiveModel,
  updateSessionModelPreservingContext,
} from '../src/lib/llm-error-recovery.js';
import type { SessionMeta } from '../src/types/agent.js';

const failedModel = 'moonshotai/Kimi-K2.6';
const fallbackModel = 'openai/gpt-5.5';

function sessionFixture(): SessionMeta {
  return {
    id: 'session-1',
    title: 'Chat 1',
    createdAt: '2026-05-31T00:00:00.000Z',
    isActive: true,
    needsAttention: false,
    model: failedModel,
    cloudProvider: 'gcp-vertex',
    trainingGoal: 'production',
    outputPolicy: 'cloud-private',
    uploadedDatasets: [
      {
        upload_id: 'upload-1',
        filename: 'hardware.md',
        format: 'md',
        source_format: 'md',
        source: 'session-upload',
        normalized_row_count: 40,
        normalized_format: 'jsonl',
        status: 'ready',
        supports_training: true,
        config_name: 'normalized',
        repo_id: 'owner/hardware',
        repo_type: 'dataset',
        normalized_path_in_repo: 'normalized/train.jsonl',
        raw_path_in_repo: 'raw/hardware.md',
        hub_url: 'https://huggingface.co/datasets/owner/hardware',
        load_dataset_snippet: 'load_dataset("owner/hardware")',
      },
    ],
  };
}

test('stale LLM error clears on model change', () => {
  const error = buildLlmErrorRecord({
    error: 'spending limit exceeded',
    errorType: 'quota_or_billing',
    model: failedModel,
    sessionId: 'session-1',
    requestId: 'request-1',
    turnId: 7,
  });

  assert.equal(clearLlmErrorOnModelChange(error, fallbackModel), null);
});

test('transient LLM error clears on new prompt', () => {
  const error = buildLlmErrorRecord({
    error: 'provider timed out',
    errorType: 'network',
    model: failedModel,
    sessionId: 'session-1',
    requestId: 'request-1',
    turnId: 7,
  });

  assert.equal(clearLlmErrorOnNewPrompt(error, 'session-1', 'request-2'), null);
});

test('error banner includes failed model name', () => {
  assert.match(
    modelSpecificErrorCopy({
      error: 'quota exceeded',
      errorType: 'quota_or_billing',
      model: failedModel,
    }),
    /moonshotai\/Kimi-K2\.6/,
  );
});

test('quota-blocked model marked unavailable', () => {
  const session = markUnavailableModelForError(sessionFixture(), {
    error: 'spending limit exceeded',
    errorType: 'quota_or_billing',
    model: failedModel,
    sessionId: 'session-1',
  });

  assert.equal(isModelUnavailable(session, failedModel), true);
});

test('switching model preserves provider, goal, output policy, and uploads', () => {
  const original = sessionFixture();
  const updated = updateSessionModelPreservingContext(original, fallbackModel);

  assert.equal(updated.model, fallbackModel);
  assert.equal(updated.cloudProvider, original.cloudProvider);
  assert.equal(updated.trainingGoal, original.trainingGoal);
  assert.equal(updated.outputPolicy, original.outputPolicy);
  assert.deepEqual(updated.uploadedDatasets, original.uploadedDatasets);
});

test('new successful request hides previous error', () => {
  const error = buildLlmErrorRecord({
    error: 'empty response',
    errorType: 'empty_response',
    model: failedModel,
    sessionId: 'session-1',
    requestId: 'request-1',
    turnId: 7,
  });

  assert.equal(clearLlmErrorOnSuccessfulRequest(error, 'session-1', 'request-2'), null);
});

test('startup health check does not show stale error for a different active model', () => {
  assert.equal(
    shouldReportLlmHealthErrorForActiveModel({ model: failedModel }, fallbackModel),
    false,
  );
  assert.equal(
    shouldReportLlmHealthErrorForActiveModel({ model: failedModel }, failedModel),
    true,
  );
});

test('select-another-model action is available in the banner UI', () => {
  const appLayoutSource = readFileSync(join(process.cwd(), 'src/components/Layout/AppLayout.tsx'), 'utf8');
  const chatInputSource = readFileSync(join(process.cwd(), 'src/components/Chat/ChatInput.tsx'), 'utf8');

  assert.match(appLayoutSource, /LLM_ERROR_SELECT_MODEL_ACTION/);
  assert.match(appLayoutSource, /shouldReportLlmHealthErrorForActiveModel/);
  assert.match(appLayoutSource, /liga-open-model-selector/);
  assert.match(chatInputSource, /addEventListener\('liga-open-model-selector'/);
});

test('stream recovery copy and request id handling are wired into the frontend', () => {
  const transportSource = readFileSync(join(process.cwd(), 'src/lib/sse-chat-transport.ts'), 'utf8');
  const hookSource = readFileSync(join(process.cwd(), 'src/hooks/useAgentChat.ts'), 'utf8');
  const agentStoreSource = readFileSync(join(process.cwd(), 'src/store/agentStore.ts'), 'utf8');

  assert.match(transportSource, /STREAM_STALL_TIMEOUT_MS\s*=\s*90_000/);
  assert.match(transportSource, /Connection stalled\. Checking session status\.\.\./);
  assert.match(transportSource, /stream_error/);
  assert.match(hookSource, /This session was lost after a server restart\. Please start a new session or retry the prompt\./);
  assert.match(hookSource, /request_id/);
  assert.match(agentStoreSource, /streamRecoveryError/);
});
