import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  REQUIRED_RESPONSE_COLUMNS,
  createResponsesQueryParams,
  createResponsesPaginationModel,
  createResponsesButtonState,
  createResponsesPanelModel,
  redactResponseText,
} from '../src/lib/responses-log-panel.js';

const appLayoutSource = readFileSync('src/components/Layout/AppLayout.tsx', 'utf8');

test('responses button is modeled as visible and clickable before or during processing', () => {
  const initial = createResponsesButtonState({ isProcessing: false, summary: null });
  const processing = createResponsesButtonState({
    isProcessing: true,
    summary: { total_responses: 0, visible_count: 0, batch_number: 1, has_rows: false, button_enabled: true },
  });

  assert.equal(initial.visible, true);
  assert.equal(initial.disabled, false);
  assert.equal(processing.visible, true);
  assert.equal(processing.disabled, false);
});

test('responses panel exposes clean empty state and required columns', () => {
  const panel = createResponsesPanelModel({ rows: [] });

  assert.equal(panel.emptyStateTitle, 'No responses yet');
  assert.match(panel.emptyStateDescription, /Fine-tuning and cloud job outcomes/);
  assert.deepEqual(REQUIRED_RESPONSE_COLUMNS.map((column) => column.label), [
    'Session Number',
    'Model Name',
    'Platform',
    'Run Type',
    'Result Storage',
    'Progress',
    'Job ID',
    'Final Artifact / Result',
  ]);
  assert.deepEqual(panel.columns.slice(0, 8), REQUIRED_RESPONSE_COLUMNS);
});

test('responses panel renders row labels with redacted artifacts', () => {
  const panel = createResponsesPanelModel({
    rows: [
      {
        display_session_number: 1,
        actual_sequence_number: 16,
        batch_number: 2,
        session_id: 'abcdef123456',
        session_title: 'Housing fine-tune',
        model_name: 'Qwen/Qwen2.5-0.5B-Instruct',
        platform: 'hf-jobs',
        run_type: 'smoke-test',
        result_storage: 'cloud-and-hf-hub',
        progress: 'completed',
        job_id: 'https://huggingface.co/jobs/acme/123',
        final_artifact_or_result: 'https://huggingface.co/acme/model?token=hf_secret_token_123',
        created_at: '2026-01-01T00:00:00+00:00',
        completed_at: '2026-01-01T00:10:00+00:00',
      },
    ],
  });

  assert.equal(panel.rows[0].cells.session, '1 (actual 16, batch 2)');
  assert.equal(panel.rows[0].cells.model, 'Qwen/Qwen2.5-0.5B-Instruct');
  assert.equal(panel.rows[0].cells.platform, 'hf-jobs');
  assert.equal(panel.rows[0].cells.progress, 'completed');
  assert.doesNotMatch(panel.rows[0].cells.result, /hf_secret/);
  assert.match(panel.rows[0].cells.result, /\[REDACTED\]/);
});

test('responses panel exposes API error state', () => {
  const panel = createResponsesPanelModel({ rows: [], error: 'Failed to load responses.' });

  assert.equal(panel.errorMessage, 'Failed to load responses.');
  assert.equal(panel.emptyStateTitle, 'No responses yet');
});

test('responses panel renders API rows when total rows are present', () => {
  const panel = createResponsesPanelModel({
    rows: [
      {
        display_session_number: 13,
        actual_sequence_number: 13,
        batch_number: 1,
        session_id: 'vertex-session',
        model_name: 'Kimi K2.6',
        platform: 'gcp-vertex',
        run_type: 'smoke-test',
        result_storage: 'cloud-private',
        progress: 'completed',
        job_id: 'projects/p/locations/us/customJobs/123',
        final_artifact_or_result: 'gs://liga-output/job-123',
      },
    ],
  });

  assert.equal(panel.rows.length, 1);
  assert.equal(panel.rows[0].cells.platform, 'gcp-vertex');
  assert.equal(panel.rows[0].cells.progress, 'completed');
  assert.equal(panel.rows[0].cells.jobId, 'projects/p/locations/us/customJobs/123');
});

test('responses pagination renders rows safely when total_pages is zero', () => {
  const pagination = createResponsesPaginationModel({
    rows: [
      {
        display_session_number: 13,
        actual_sequence_number: 13,
        batch_number: 1,
        session_id: 'vertex-session',
        model_name: 'Kimi K2.6',
        platform: 'gcp-vertex',
        run_type: 'smoke-test',
        result_storage: 'cloud-private',
        progress: 'completed',
        job_id: 'projects/p/locations/us/customJobs/123',
        final_artifact_or_result: 'gs://liga-output/job-123',
      },
    ],
    page: 1,
    page_size: 50,
    total_rows: 1,
    total_pages: 0,
    has_next: false,
    has_previous: false,
  });

  assert.equal(pagination.label, 'Page 1 of 1 • 1 responses');
});

test('responses pagination renders rows safely when total_pages is missing', () => {
  const pagination = createResponsesPaginationModel({
    rows: [
      {
        display_session_number: 13,
        actual_sequence_number: 13,
        batch_number: 1,
        session_id: 'vertex-session',
        model_name: 'Kimi K2.6',
        platform: 'gcp-vertex',
        run_type: 'smoke-test',
        result_storage: 'cloud-private',
        progress: 'completed',
        job_id: 'projects/p/locations/us/customJobs/123',
        final_artifact_or_result: 'gs://liga-output/job-123',
      },
    ],
    page: 1,
    page_size: 50,
    total_rows: 1,
    total_pages: undefined as unknown as number,
    has_next: false,
    has_previous: false,
  });

  assert.equal(pagination.label, 'Page 1 of 1 • 1 responses');
});

test('responses pagination avoids transient zero label while loading', () => {
  const pagination = createResponsesPaginationModel(null, { loading: true });

  assert.equal(pagination.label, 'Loading responses...');
  assert.equal(pagination.canGoPrevious, false);
  assert.equal(pagination.canGoNext, false);
});

test('responses dialog model renders loading then terminal HF row', () => {
  const loading = createResponsesPaginationModel(null, { loading: true });
  const panel = createResponsesPanelModel({
    rows: [
      {
        display_session_number: 11,
        actual_sequence_number: 11,
        batch_number: 1,
        session_id: '00a5ec95-6130-4ea4-9d72-c6b26116e051',
        model_name: 'moonshotai/Kimi-K2.6',
        platform: 'hf-jobs',
        run_type: 'smoke-test',
        result_storage: 'hf-hub',
        progress: 'completed',
        job_id: 'https://huggingface.co/jobs/ligaments-dev/6a277fd3ece949d7b3dcc4db',
        final_artifact_or_result: 'https://huggingface.co/ligaments-dev/gst-qwen2.5-0.5b-sft-smoke',
        completed_at: '2026-06-09T02:52:03.051000+00:00',
      },
    ],
  });

  assert.equal(loading.label, 'Loading responses...');
  assert.equal(panel.rows[0].cells.progress, 'completed');
  assert.equal(panel.rows[0].cells.storage, 'hf-hub');
  assert.equal(panel.rows[0].cells.runType, 'smoke-test');
  assert.match(panel.rows[0].cells.result, /gst-qwen2\.5-0\.5b-sft-smoke/);
  assert.notEqual(panel.rows[0].cells.completedAt, '-');
});

test('responses panel distinguishes filtered empty state from global no-data state', () => {
  const empty = createResponsesPanelModel({ rows: [] });
  const filtered = createResponsesPanelModel({ rows: [], filtersActive: true });

  assert.equal(empty.emptyStateTitle, 'No responses yet');
  assert.equal(filtered.emptyStateTitle, 'No responses match your filters');
});

test('response redaction handles bearer and hf tokens', () => {
  assert.equal(redactResponseText('Authorization: Bearer hf_secret_token_123'), 'Authorization: Bearer [REDACTED]');
  assert.equal(redactResponseText('model?token=hf_secret_token_123'), 'model?token=[REDACTED]');
});

test('app layout includes the responses button component', () => {
  assert.match(appLayoutSource, /ResponsesLogButton/);
});

test('responses query params include pagination and omit empty filters', () => {
  const params = createResponsesQueryParams({
    page: 2,
    pageSize: 25,
    platform: 'hf-jobs',
    progress: 'completed',
    model: '',
    jobId: 'job-123',
    q: 'housing',
  });

  assert.equal(
    params.toString(),
    'page=2&page_size=25&platform=hf-jobs&progress=completed&job_id=job-123&q=housing',
  );
});

test('responses pagination model exposes previous and next controls', () => {
  const pagination = createResponsesPaginationModel({
    rows: [],
    page: 2,
    page_size: 50,
    total_rows: 101,
    total_pages: 3,
    has_next: true,
    has_previous: true,
  });

  assert.equal(pagination.label, 'Page 2 of 3 • 101 responses');
  assert.equal(pagination.canGoPrevious, true);
  assert.equal(pagination.canGoNext, true);
  assert.equal(pagination.previousPage, 1);
  assert.equal(pagination.nextPage, 3);
});

test('responses dialog source does not clear fetched rows on close', () => {
  const buttonSource = readFileSync('src/components/ResponsesLogButton.tsx', 'utf8');

  assert.match(buttonSource, /setOpen\(false\)/);
  assert.doesNotMatch(buttonSource, /setRows\(\[\]\)/);
});

test('chat hydration clears stale processing when backend is terminal', () => {
  const hookSource = readFileSync('src/hooks/useAgentChat.ts', 'utf8');

  assert.match(hookSource, /backendIsProcessing/);
  assert.match(hookSource, /updateSession\(sessionId,\s*\{\s*isProcessing:\s*false,\s*activityStatus:\s*\{\s*type:\s*'idle'\s*\}/s);
  assert.match(hookSource, /fresh\.info && !fresh\.info\.is_processing/);
});
