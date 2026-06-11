import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  PREFLIGHT_ACTION_COPY,
  buildManualPreflightRequest,
  createManualPreflightNotRunMarkdown,
  loadPersistedTrainingPreflight,
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
        output_policy: 'cloud-and-hf-hub',
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
    output_policy: 'cloud-and-hf-hub',
  });
  assert.deepEqual(request.dataset_summary, { rows: 10 });
  assert.equal(request.include_fallbacks, true);
  assert.equal(request.force_refresh, false);
  assert.equal(request.timeout_seconds, 15);
});

test('manual refresh request sets force refresh true', () => {
  const request = buildManualPreflightRequest({
    sessionId: 's1',
    runId: 'run1',
    plannerOutput: { recommendation: { selected_provider: { provider_id: 'hf-jobs' } } },
    forceRefresh: true,
  });

  assert.equal(request.session_id, 's1');
  assert.equal(request.run_id, 'run1');
  assert.equal(request.force_refresh, true);
  assert.equal(request.include_fallbacks, true);
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

test('manual preflight request sends full planner recommendation when available', () => {
  const request = buildManualPreflightRequest({
    sessionId: 's1',
    plannerOutput: {
      provider: 'gcp-vertex',
      recommended_model: 'Qwen/Qwen2.5-0.5B-Instruct',
      hardware_id: 'gcp-vertex:n1-standard-8-t4',
      output_policy: 'cloud-private',
      recommendation: {
        selected_provider: { provider_id: 'gcp-vertex' },
        selected_model: { model_id: 'Qwen/Qwen2.5-0.5B-Instruct' },
        selected_hardware: { hardware_id: 'gcp-vertex:n1-standard-8-t4' },
        output_policy: 'cloud-private',
      },
    },
  });

  assert.equal(request.recommendation?.provider, 'gcp-vertex');
  assert.equal(request.recommendation?.recommended_model, 'Qwen/Qwen2.5-0.5B-Instruct');
  assert.equal(request.recommendation?.hardware_id, 'gcp-vertex:n1-standard-8-t4');
  assert.equal(request.recommendation?.output_policy, 'cloud-private');
});

test('manual preflight request omits incomplete unknown recommendation', () => {
  const request = buildManualPreflightRequest({
    sessionId: 's1',
    plannerOutput: {
      recommendation: {
        selected_provider: { provider_id: 'unknown' },
        selected_model: { model_id: 'unknown' },
        selected_hardware: { hardware_id: null },
      },
    },
  });

  assert.equal('recommendation' in request, false);
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

test('persisted preflight load uses GET only and renders unknown result', async () => {
  const calls: string[] = [];
  const states: ManualPreflightState[] = [];

  await loadPersistedTrainingPreflight({
    sessionId: 's1',
    getLatest: async (sessionId) => {
      calls.push(`GET session ${sessionId}`);
      return preflightResult();
    },
    getRun: async () => {
      throw new Error('unexpected run fetch');
    },
    onStateChange: (next) => states.push(next),
  });

  assert.deepEqual(calls, ['GET session s1']);
  assert.equal(states[0].status, 'loading');
  assert.equal(states.at(-1)?.status, 'success');
  assert.equal(states.at(-1)?.result?.launch_ready, false);
  assert.match(states.at(-1)?.markdown ?? '', /Stored preflight may be stale/);
  assert.match(states.at(-1)?.markdown ?? '', /Updated at: 2026-06-09T08:00:00Z/);
  assert.doesNotMatch(states.at(-1)?.markdown ?? '', /status: unknown \| Passed/i);
});

test('persisted run preflight load uses run GET when run id exists', async () => {
  const calls: string[] = [];

  await loadPersistedTrainingPreflight({
    sessionId: 's1',
    runId: 'run1',
    getLatest: async () => {
      throw new Error('unexpected session fetch');
    },
    getRun: async (sessionId, runId) => {
      calls.push(`GET run ${sessionId}/${runId}`);
      return preflightResult({ run_id: runId });
    },
    onStateChange: () => undefined,
  });

  assert.deepEqual(calls, ['GET run s1/run1']);
});

test('persisted preflight 404 null shows not-run state without scary error', async () => {
  const states: ManualPreflightState[] = [];

  await loadPersistedTrainingPreflight({
    sessionId: 's1',
    getLatest: async () => null,
    getRun: async () => null,
    onStateChange: (next) => states.push(next),
  });

  const finalState = states.at(-1);
  assert.equal(finalState?.status, 'not_run');
  assert.match(finalState?.markdown ?? '', /No preflight has been run yet/);
  assert.doesNotMatch(finalState?.markdown ?? '', /Error|failed/i);
});

test('persisted passed preflight renders launch ready and timestamps', async () => {
  const states: ManualPreflightState[] = [];

  await loadPersistedTrainingPreflight({
    sessionId: 's1',
    getLatest: async () => preflightResult({ status: 'passed', launch_ready: true, safe_summary: 'Preflight passed.' }),
    getRun: async () => null,
    onStateChange: (next) => states.push(next),
  });

  const finalState = states.at(-1);
  assert.equal(finalState?.status, 'success');
  assert.equal(finalState?.result?.launch_ready, true);
  assert.match(finalState?.markdown ?? '', /Launch ready: yes/);
  assert.match(finalState?.markdown ?? '', /Created at: 2026-06-09T08:00:00Z/);
  assert.match(finalState?.markdown ?? '', /Launch still requires explicit approval/);
});

test('persisted preflight load error is redacted and does not block planner', async () => {
  const fakeToken = `hf_${'B'.repeat(35)}`;
  const states: ManualPreflightState[] = [];

  await loadPersistedTrainingPreflight({
    sessionId: 's1',
    getLatest: async () => {
      throw new Error(`GET failed ${fakeToken}`);
    },
    getRun: async () => null,
    onStateChange: (next) => states.push(next),
  });

  const finalState = states.at(-1);
  assert.equal(finalState?.status, 'error');
  assert.doesNotMatch(finalState?.error ?? '', /hf_[A-Za-z0-9]/);
  assert.match(finalState?.error ?? '', /\[REDACTED\]/);
  assert.match(finalState?.markdown ?? '', /Static recommendation remains visible/);
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

test('manual refresh replaces stale displayed result', async () => {
  const states: ManualPreflightState[] = [{ status: 'success', result: preflightResult(), markdown: 'stale' }];
  const request = buildManualPreflightRequest({ sessionId: 's1', forceRefresh: true });

  await runManualTrainingPreflight({
    request,
    run: async (actualRequest) => {
      assert.equal(actualRequest.force_refresh, true);
      return preflightResult({ status: 'passed', launch_ready: true, safe_summary: 'Fresh preflight passed.' });
    },
    onStateChange: (next) => states.push(next),
  });

  assert.equal(states.at(-1)?.status, 'success');
  assert.equal(states.at(-1)?.result?.launch_ready, true);
  assert.match(states.at(-1)?.markdown ?? '', /Fresh preflight passed/);
  assert.doesNotMatch(states.at(-1)?.markdown ?? '', /^stale$/);
});

test('tool group exposes a manual preflight button without auto-run', () => {
  assert.match(toolCallGroupSource, /Run preflight check/);
  assert.match(toolCallGroupSource, /Refresh preflight/);
  assert.match(toolCallGroupSource, /loadPersistedTrainingPreflight/);
  assert.match(toolCallGroupSource, /handleRunPreflight/);
  assert.doesNotMatch(toolCallGroupSource, /useEffect\([\s\S]{0,240}runTrainingPreflight/);
  assert.doesNotMatch(toolCallGroupSource, /useEffect\([\s\S]{0,240}force_refresh:\s*true/);
});

test('manual preflight panel writes use the idempotent panel guard', () => {
  assert.match(toolCallGroupSource, /setPanelIfChanged/);
  assert.doesNotMatch(
    toolCallGroupSource,
    /setPanel\(\{\s*title:\s*'Training Preflight'/,
  );
});

test('manual action copy avoids provider job launch wording', () => {
  assert.match(PREFLIGHT_ACTION_COPY.noJobs, /No provider jobs will be launched/);
  assert.match(PREFLIGHT_ACTION_COPY.notLaunch, /not a training launch/);
  assert.doesNotMatch(PREFLIGHT_ACTION_COPY.button, /launch/i);
});
