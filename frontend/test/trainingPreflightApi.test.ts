import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

import {
  getLatestSessionPreflight,
  getRunPreflight,
  runTrainingPreflight,
} from '../src/lib/training-preflight-api.js';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function preflightBody() {
  return {
    preflight_id: 'pf1',
    session_id: 's1',
    run_id: 'r1',
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
    blocking_reasons: [],
    warning_reasons: [],
    unknown_reasons: ['Live checks are not implemented.'],
    safe_summary: 'Training preflight unknown.',
    cache: { hit: false },
    metadata: { provider_jobs_launched: false, resources_created: false },
  };
}

test('runTrainingPreflight posts to training preflight route with safe JSON', async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  globalThis.fetch = async (input, init) => {
    calls.push({ url: String(input), init });
    return jsonResponse(preflightBody());
  };

  const result = await runTrainingPreflight({
    session_id: 's1',
    run_id: 'r1',
    recommendation: { provider: 'hf-jobs' },
    dataset_summary: { rows: 10 },
    target_namespace: 'alice',
    target_repo_id: 'alice/model',
    include_fallbacks: true,
    force_refresh: true,
    timeout_seconds: 10,
  });

  assert.equal(result?.preflight_id, 'pf1');
  assert.equal(calls[0].url, '/api/training-preflight');
  assert.equal(calls[0].init?.method, 'POST');
  const body = JSON.parse(String(calls[0].init?.body));
  assert.equal(body.session_id, 's1');
  assert.equal(body.timeout_seconds, 10);
  assert.doesNotMatch(JSON.stringify(body), /hf_[A-Za-z0-9]/);
});

test('preflight getters use session and run routes', async () => {
  const urls: string[] = [];
  globalThis.fetch = async (input) => {
    urls.push(String(input));
    return jsonResponse(preflightBody());
  };

  await getLatestSessionPreflight('s1');
  await getRunPreflight('s1', 'r1');

  assert.deepEqual(urls, [
    '/api/session/s1/preflight',
    '/api/session/s1/runs/r1/preflight',
  ]);
});

test('latest preflight 404 returns null', async () => {
  globalThis.fetch = async () => jsonResponse({ detail: 'not found' }, 404);

  assert.equal(await getLatestSessionPreflight('missing'), null);
  assert.equal(await getRunPreflight('missing', 'r1'), null);
});

test('non-404 failures throw redacted safe errors', async () => {
  globalThis.fetch = async () =>
    jsonResponse({ detail: `HF_TOKEN=hf_${'A'.repeat(35)}` }, 500);

  await assert.rejects(
    () => getLatestSessionPreflight('s1'),
    (error) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /Training preflight API returned 500/);
      assert.doesNotMatch(error.message, /hf_[A-Za-z0-9]/);
      assert.match(error.message, /\[REDACTED\]/);
      return true;
    },
  );
});
