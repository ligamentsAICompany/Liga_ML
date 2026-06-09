import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  auditEventTitle,
  buildAuditFilters,
  chronologicalAuditEvents,
  eventContainsSecret,
  filterAuditEvents,
  safeAuditLinks,
  severityColor,
  timelineEmptyMessage,
} from '../src/lib/audit-timeline.js';
import type { AuditEvent } from '../src/types/audit.js';

const events: AuditEvent[] = [
  {
    audit_id: 'a2',
    session_id: 's1',
    run_id: 'r1',
    event_type: 'provider_job_failed',
    category: 'provider_job',
    severity: 'error',
    status: 'failed',
    title: 'AWS SageMaker job failed',
    message: 'Provider reported quota blocked.',
    timestamp: '2026-06-07T10:02:00Z',
    actor: 'provider',
    provider: 'aws-sagemaker',
    job_url: 'https://console.aws.amazon.com/sagemaker/train-1',
    artifact_url: 'javascript:alert(1)',
    safe_metadata: {},
  },
  {
    audit_id: 'a1',
    session_id: 's1',
    run_id: 'r1',
    event_type: 'approval_required',
    category: 'approval',
    severity: 'warning',
    status: 'pending',
    title: 'Approval required',
    message: 'Review AWS SageMaker job before launch.',
    timestamp: '2026-06-07T10:01:00Z',
    actor: 'assistant',
    provider: 'aws-sagemaker',
    approval_id: 'approval-1',
    safe_metadata: {},
  },
  {
    audit_id: 'a3',
    session_id: 's1',
    event_type: 'dataset_upload_succeeded',
    category: 'dataset',
    severity: 'info',
    status: 'succeeded',
    title: '',
    message: 'Dataset uploaded.',
    timestamp: '2026-06-07T10:00:00Z',
    actor: 'system',
    provider: 'hf-jobs',
    dataset_name: 'safe.csv',
    safe_metadata: { rows: 3 },
  },
];

test('audit timeline sorts events chronologically', () => {
  const ordered = chronologicalAuditEvents(events);

  assert.deepEqual(ordered.map((event) => event.audit_id), ['a3', 'a1', 'a2']);
});

test('audit timeline filters provider category and severity', () => {
  const filtered = filterAuditEvents(events, {
    provider: 'aws-sagemaker',
    category: 'approval',
    severity: 'warning',
  });

  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].event_type, 'approval_required');
});

test('audit timeline builds available filter values', () => {
  const filters = buildAuditFilters(events);

  assert.deepEqual(filters.providers, ['aws-sagemaker', 'hf-jobs']);
  assert.deepEqual(filters.categories, ['approval', 'dataset', 'provider_job']);
  assert.deepEqual(filters.severities, ['error', 'info', 'warning']);
});

test('audit timeline exposes severity styling and empty state', () => {
  assert.equal(severityColor('error'), 'error');
  assert.equal(severityColor('critical'), 'error');
  assert.equal(severityColor('warning'), 'warning');
  assert.match(timelineEmptyMessage({ provider: 'hf-jobs' }), /No audit events match/);
  assert.match(timelineEmptyMessage({}), /No audit events yet/);
});

test('audit timeline uses safe links only', () => {
  const links = safeAuditLinks(events[0]);

  assert.deepEqual(links, [{ label: 'Job', href: 'https://console.aws.amazon.com/sagemaker/train-1' }]);
});

test('audit timeline has readable fallback titles', () => {
  assert.equal(auditEventTitle(events[2]), 'dataset upload succeeded');
});

test('audit timeline detects secret-like values before rendering', () => {
  assert.equal(eventContainsSecret(events[0]), false);
  assert.equal(
    eventContainsSecret({
      ...events[2],
      message: 'token=hf_secret',
    }),
    true,
  );
});
