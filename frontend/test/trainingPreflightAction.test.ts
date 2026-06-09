import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  PREFLIGHT_ACTION_COPY,
  buildManualPreflightRequest,
  createManualPreflightNotRunMarkdown,
  runManualTrainingPreflight,
  type ManualPreflightState,
} from '../src/lib/training-preflight-action.js';
import type { TrainingPreflightResult } from '../src/types/agent.js';

const toolCallGroupSource = readFileSync('src/components/Chat/ToolCallGroup.tsx', 'utf8');

function preflightResult(overrides: Partial<TrainingPreflightResult> = {}): TrainingPreflightResult {
  return {
    preflight_id: 'pf1',
    session_id: 's1',
    run_id: 'run1',
    created_at: '2026-06-09T08:00:00Z',
    updated_at: '2026-06-09T08:00:00Z',
    status: 'unknown',
    launch_ready: false,
    provider: 'hf-jobs',
    model_id: 'Qwen/Qwen2.5-0.5B-Instruct',
    hardware_id: 'hf-jobs:t4-small',
    output_policy: 'cloud-and-hf-hub',
    primary: {
      provider: 'hf-jobs',
      status: 'unknown',
      launch_ready: false,
      checks: [],
      blocking_reasons: [],
      warning_reasons: [],
      unknown_reasons: ['Live checks are not implemented.'],
      metadata: {},
    },
    fallbacks: [],
    verified_recommendation: null,
    blocking_reasons: [],
    warning_reasons: [],
    unknown_reasons: ['Live checks are not implemented.'],
    safe_summary: 'Local-only preflight result remains unknown.',
    cache: { hit: false },
    metadata: {
      provider_jobs_launched: false,
      resources_created: false,
      live_checks_optional: true,
    },
    ...overrides,
  };
}

test('manual preflight request uses planner context and safe defaults', () => {
  const request = buildManualPreflightRequest({
    sessionId: 's1',
    runId: 'run1',
    plannerOutput: {
      recommendation: {
        selected_provider: { provider_id: 'hf-jobs' },
        selected_model: { model_id: 'Qwen/Qwen2.5-0.5B-Instruct' },
        selected_hardware: { hardware_id: 'hf-jobs:t4-small' },
      },
      datasetSummary: { rows: 10 },
    },
  });

  assert.equal(request.session_id, 's1');
  assert.equal(request.run_id, 'run1');
  assert.deepEqual(request.recommendation, {
    selected_provider: { provider_id: 'hf-jobs' },
    selected_model: { model_id: 'Qwen/Qwen2.5-0.5B-Instruct' },
    selected_hardware: { hardware_id: 'hf-jobs:t4-small' },
  });
  assert.deepEqual(request.dataset_summary, { rows: 10 });
  assert.equal(request.include_fallbacks, true);
  assert.equal(request.force_refresh, false);
  assert.equal(request.timeout_seconds, 15);
});

test('manual preflight request falls back to session-only when recommendation missing', () => {
  const request = buildManualPreflightRequest({ sessionId: 's1' });

  assert.deepEqual(request, {
    session_id: 's1',
    include_fallbacks: true,
    force_refresh: false,
    timeout_seconds: 15,
  });
});

test('manual preflight not-run copy is safety-focused', () => {
  const markdown = createManualPreflightNotRunMarkdown();

  assert.match(markdown, /Static recommendation is not the same as verified launch readiness/);
  assert.match(markdown, /This is a preflight check, not a training launch/);
  assert.match(markdown, /No provider jobs will be launched/);
  assert.match(markdown, /No resources will be created/);
  assert.match(markdown, /Unknown does not mean passed/);
  assert.match(markdown, /Launch still requires explicit approval/);
  assert.doesNotMatch(markdown, /launch-ready static recommendation/i);
});

test('manual preflight action does not run until clicked helper is called', async () => {
  let calls = 0;
  const state: ManualPreflightState = { status: 'not_run' };

  assert.equal(calls, 0);
  assert.equal(state.status, 'not_run');

  await runManualTrainingPreflight({
    request: { session_id: 's1', include_fallbacks: true },
    run: async () => {
      calls += 1;
      return preflightResult();
    },
    onStateChange: (next) => Object.assign(state, next),
  });

  assert.equal(calls, 1);
  assert.equal(state.status, 'success');
  assert.equal(state.result?.launch_ready, false);
  assert.match(state.markdown ?? '', /Launch ready: no/);
  assert.match(state.markdown ?? '', /Unknown does not mean passed/);
});

test('manual preflight action exposes loading state and disables repeated clicks', async () => {
  const states: ManualPreflightState[] = [];
  await runManualTrainingPreflight({
    request: { session_id: 's1' },
    run: async () => preflightResult({ status: 'passed', launch_ready: true, safe_summary: 'Preflight passed.' }),
    onStateChange: (next) => states.push(next),
  });

  assert.equal(states[0].status, 'checking');
  assert.equal(states[0].disabled, true);
  assert.match(states[0].markdown ?? '', /Preflight is running/);
  assert.equal(states.at(-1)?.status, 'success');
  assert.equal(states.at(-1)?.result?.launch_ready, true);
  assert.match(states.at(-1)?.markdown ?? '', /Launch ready: yes/);
});

test('manual preflight action redacts safe retry error state', async () => {
  const fakeToken = `hf_${'A'.repeat(35)}`;
  const states: ManualPreflightState[] = [];

  await runManualTrainingPreflight({
    request: { session_id: 's1' },
    run: async () => {
      throw new Error(`provider failed with token ${fakeToken}`);
    },
    onStateChange: (next) => states.push(next),
  });

  const finalState = states.at(-1);
  assert.equal(finalState?.status, 'error');
  assert.match(finalState?.markdown ?? '', /Run preflight check/);
  assert.doesNotMatch(finalState?.error ?? '', /hf_[A-Za-z0-9]/);
  assert.match(finalState?.error ?? '', /\[REDACTED\]/);
});

test('tool group exposes a manual preflight button without auto-run', () => {
  assert.match(toolCallGroupSource, /Run preflight check/);
  assert.match(toolCallGroupSource, /handleRunPreflight/);
  assert.doesNotMatch(toolCallGroupSource, /useEffect\([\s\S]{0,240}runTrainingPreflight/);
});

test('manual action copy avoids provider job launch wording', () => {
  assert.match(PREFLIGHT_ACTION_COPY.noJobs, /No provider jobs will be launched/);
  assert.match(PREFLIGHT_ACTION_COPY.notLaunch, /not a training launch/);
  assert.doesNotMatch(PREFLIGHT_ACTION_COPY.button, /launch/i);
});
