import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import { createTrainingPreflightPanel } from '../src/lib/training-preflight-panel.js';
import type { TrainingPreflightResult } from '../src/types/agent.js';

const toolCallGroupSource = readFileSync('src/components/Chat/ToolCallGroup.tsx', 'utf8');

function result(overrides: Partial<TrainingPreflightResult> = {}): TrainingPreflightResult {
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
      blocking_reasons: [],
      warning_reasons: ['Dataset context was not available.'],
      unknown_reasons: ['Live provider credential checks are not implemented in this slice.'],
      metadata: {},
      checks: [
        {
          check_id: 'static_recommendation',
          provider: 'hf-jobs',
          category: 'metadata',
          label: 'Static recommendation',
          status: 'passed',
          severity: 'info',
          message: 'Static training recommendation is present.',
          details: { required: true },
          docs_verification_required: false,
        },
        {
          check_id: 'provider_credentials_live',
          provider: 'hf-jobs',
          category: 'credentials',
          label: 'Provider credentials',
          status: 'unknown',
          severity: 'error',
          message: 'Live provider credential checks are not implemented in this slice.',
          details: { required: true },
          error_code: 'live_probe_not_implemented',
          docs_verification_required: true,
        },
        {
          check_id: 'provider_job_launch',
          provider: 'hf-jobs',
          category: 'safety',
          label: 'Provider job launch',
          status: 'skipped',
          severity: 'info',
          message: 'Provider job launch is intentionally skipped for local preflight.',
          details: { required: false, applicable: false },
          docs_verification_required: false,
        },
      ],
    },
    fallbacks: [],
    verified_recommendation: {
      recommendation: {
        selected_provider: { provider_id: 'hf-jobs', display_name: 'Hugging Face Jobs' },
        selected_model: { model_id: 'Qwen/Qwen2.5-0.5B-Instruct' },
        selected_hardware: { hardware_id: 'hf-jobs:t4-small' },
      },
    },
    blocking_reasons: [],
    warning_reasons: ['Dataset context was not available.'],
    unknown_reasons: ['Live provider credential checks are not implemented in this slice.'],
    safe_summary: 'Training preflight unknown; launch_ready=false.',
    cache: { hit: false },
    metadata: {
      provider_jobs_launched: false,
      resources_created: false,
      live_checks_optional: true,
      mode: 'local_non_network',
    },
    ...overrides,
  };
}

test('preflight panel renders not-run and checking states safely', () => {
  const notRun = createTrainingPreflightPanel(result({ status: 'not_run', safe_summary: '', primary: { ...result().primary, status: 'not_run', checks: [] } }));
  const checking = createTrainingPreflightPanel(result({ status: 'checking', primary: { ...result().primary, status: 'checking' } }));

  assert.match(notRun.markdown, /Preflight not run/);
  assert.match(checking.markdown, /Preflight is checking/);
  assert.doesNotMatch(notRun.markdown, /undefined|null/);
});

test('preflight panel renders passed launch-ready state without confusing static recommendation', () => {
  const panel = createTrainingPreflightPanel(result({
    status: 'passed',
    launch_ready: true,
    primary: { ...result().primary, status: 'passed', launch_ready: true, unknown_reasons: [], warning_reasons: [] },
    unknown_reasons: [],
    warning_reasons: [],
    safe_summary: 'Training preflight passed; launch_ready=true.',
  }));

  assert.match(panel.markdown, /Launch ready: yes/);
  assert.match(panel.markdown, /Launch still requires explicit approval/);
  assert.match(panel.markdown, /Static recommendation and preflight verification are different/);
});

test('preflight panel renders warning failed unknown and skipped statuses correctly', () => {
  const warning = createTrainingPreflightPanel(result({ status: 'warning', launch_ready: true }));
  const failed = createTrainingPreflightPanel(result({
    status: 'failed',
    blocking_reasons: ['Static model size exceeds hardware memory.'],
    primary: { ...result().primary, status: 'failed', blocking_reasons: ['Static model size exceeds hardware memory.'] },
  }));
  const unknown = createTrainingPreflightPanel(result());

  assert.match(warning.markdown, /Ready with warnings or review required/);
  assert.match(failed.markdown, /Not launch-ready/);
  assert.match(failed.markdown, /Static model size exceeds hardware memory/);
  assert.match(unknown.markdown, /Not proven \/ not launch-ready by default/);
  assert.match(unknown.markdown, /Provider job launch.*skipped.*not applicable/);
  assert.doesNotMatch(unknown.markdown, /status: unknown \| Passed/i);
});

test('preflight panel renders reasons provider model hardware output policy and safety metadata', () => {
  const panel = createTrainingPreflightPanel(result());

  assert.match(panel.markdown, /Provider ID: hf-jobs/);
  assert.match(panel.markdown, /Model: Qwen\/Qwen2\.5-0\.5B-Instruct/);
  assert.match(panel.markdown, /Hardware: hf-jobs:t4-small/);
  assert.match(panel.markdown, /Output policy: cloud-and-hf-hub/);
  assert.match(panel.markdown, /Blocking Reasons/);
  assert.match(panel.markdown, /Warnings/);
  assert.match(panel.markdown, /Unknowns/);
  assert.match(panel.markdown, /No provider jobs were launched: true/);
  assert.match(panel.markdown, /No resources were created: true/);
});

test('preflight panel renders verified fallback and cache info', () => {
  const panel = createTrainingPreflightPanel(result({
    fallbacks: [
      {
        fallback_id: 'fb1',
        provider: 'hf-jobs',
        model_id: 'Qwen/Qwen2.5-1.5B-Instruct',
        hardware_id: 'hf-jobs:a10g-small',
        status: 'warning',
        launch_ready: false,
        checks: [],
        reason: 'Use larger hardware after review.',
        metadata: {},
      },
    ],
    cache: {
      cache_key: 'session:s1:preflight',
      hit: true,
      ttl_seconds: 300,
      created_at: '2026-06-09T08:00:00Z',
      expires_at: '2026-06-09T08:05:00Z',
    },
  }));

  assert.match(panel.markdown, /Fallbacks/);
  assert.match(panel.markdown, /Qwen\/Qwen2\.5-1\.5B-Instruct/);
  assert.match(panel.markdown, /Cache Info/);
  assert.match(panel.markdown, /cache hit: true/i);
});

test('preflight panel redacts secrets and signed URLs', () => {
  const fakeAwsKey = `AKIA${'A'.repeat(16)}`;
  const privateKey = `-----BEGIN ${'PRIVATE'} KEY-----\nabc\n-----END ${'PRIVATE'} KEY-----`;
  const mongoUri = `mongodb+srv://${'user'}:${'pass'}@example.mongodb.net/db`;
  const signedUrl = `https://example.com/file?${'X-Amz-Signature'}=abc123&${'X-Amz-Credential'}=secret`;
  const panel = createTrainingPreflightPanel(result({
    safe_summary: [
      `HF_TOKEN=hf_${'A'.repeat(35)}`,
      `AWS key ${fakeAwsKey}`,
      `mongo=${mongoUri}`,
      privateKey,
      signedUrl,
    ].join(' '),
  }));

  assert.doesNotMatch(panel.markdown, /hf_[A-Za-z0-9]/);
  assert.doesNotMatch(panel.markdown, new RegExp(fakeAwsKey));
  assert.doesNotMatch(panel.markdown, /user:pass@/);
  assert.doesNotMatch(panel.markdown, /BEGIN PRIVATE KEY/);
  assert.doesNotMatch(panel.markdown, /X-Amz-Signature=abc123/);
  assert.match(panel.markdown, /\[REDACTED\]/);
});

test('tool panel source recognizes training preflight without auto-running it', () => {
  assert.match(toolCallGroupSource, /training_preflight/);
  assert.doesNotMatch(toolCallGroupSource, /runTrainingPreflight\(/);
});
