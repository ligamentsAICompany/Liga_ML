import assert from 'node:assert/strict';
import { test } from 'node:test';

import { eventContainsSecret, filterAuditEvents } from '../src/lib/audit-timeline.js';
import { buildProviderCards, usageEntryTitle } from '../src/lib/usage-dashboard.js';
import { containsSecretLikeValue, redactJsonLike, redactText } from '../src/lib/redaction.js';
import { parseLigaTrainingResult } from '../src/utils/trainingResult.js';
import type { AuditEvent } from '../src/types/audit.js';
import type { UsageSummary } from '../src/types/usage.js';

test('redacts secret-like frontend text and JSON without hiding normal artifact URLs', () => {
  const redacted = redactText([
    `HF_TOKEN=${'hf_' + 'A'.repeat(35)}`,
    'HF_TOKEN=hf_FAKE_TEST_TOKEN_1234567890',
    'AWS_SECRET_ACCESS_KEY=FAKEAWSSECRET1234567890',
    `Authorization: Bearer ${'b'.repeat(32)}`,
    'mongodb+srv://user:pass@example.mongodb.net/db',
    'MONGODB_URI=mongodb+srv://fake_user:fake_password@example.mongodb.net/test',
    'PRIVATE_KEY=-----BEGIN PRIVATE KEY-----FAKE-----END PRIVATE KEY-----',
    's3://bucket/model.tar.gz',
    'gs://bucket/path',
    'https://huggingface.co/alice/model',
  ].join('\n'));

  assert.doesNotMatch(redacted, /hf_[A-Za-z0-9]/);
  assert.doesNotMatch(redacted, /FAKEAWSSECRET/);
  assert.doesNotMatch(redacted, /Bearer b/);
  assert.doesNotMatch(redacted, /user:pass@/);
  assert.doesNotMatch(redacted, /fake_password/);
  assert.doesNotMatch(redacted, /BEGIN PRIVATE KEY/);
  assert.match(redacted, /\[REDACTED\]/);
  assert.match(redacted, /s3:\/\/bucket\/model\.tar\.gz/);
  assert.match(redacted, /gs:\/\/bucket\/path/);
  assert.match(redacted, /https:\/\/huggingface\.co\/alice\/model/);

  assert.deepEqual(redactJsonLike({ OPENAI_API_KEY: `sk-${'c'.repeat(45)}`, safe: 'ok' }), {
    OPENAI_API_KEY: '[REDACTED]',
    safe: 'ok',
  });
});

test('audit, usage, and training helpers redact fake secrets before rendering', () => {
  const auditEvent: AuditEvent = {
    audit_id: 'a1',
    session_id: 's1',
    event_type: 'provider_error',
    provider: 'aws-sagemaker',
    category: 'error',
    severity: 'error',
    status: 'failed',
    title: `failed with sk-${'a'.repeat(45)}`,
    message: `Authorization: Bearer ${'b'.repeat(32)}`,
    actor: 'provider',
    safe_metadata: { token: `hf_${'C'.repeat(35)}` },
  };
  const [safeAudit] = filterAuditEvents([auditEvent], {});
  assert.equal(eventContainsSecret(auditEvent), true);
  assert.equal(containsSecretLikeValue(safeAudit), false);
  assert.doesNotMatch(JSON.stringify(safeAudit), /hf_|sk-|Bearer b/);

  const summary: UsageSummary = {
    total_estimated_cost_usd: 0,
    total_known_cost_usd: 0,
    cost_by_provider: {},
    cost_by_session: {},
    cost_by_run: {},
    recent_usage_entries: [{
      usage_id: 'u1',
      session_id: 's1',
      provider: 'hf-jobs',
      operation: 'run',
      job_id: `job-${'hf_' + 'D'.repeat(35)}`,
      artifact_url: 'https://huggingface.co/alice/model',
      status: 'running',
      currency: 'USD',
      cost_source: 'unknown',
      cost_confidence: 'unknown',
      approved: true,
      quota_status: 'unknown',
    }],
    quota_warnings: [{ provider: 'hf-jobs', message: `token=${'hf_' + 'E'.repeat(35)}`, usage_id: 'u1' }],
    budget_warnings: [],
    provider_readiness: { hf_jobs: { configured: true, notes: [`OPENAI_API_KEY=sk-${'f'.repeat(45)}`] } },
  };
  const [hfCard] = buildProviderCards(summary);
  assert.equal(usageEntryTitle(summary.recent_usage_entries[0]).includes('hf_'), false);
  assert.equal(hfCard.recentJobs[0].artifact_url, 'https://huggingface.co/alice/model');
  assert.doesNotMatch(JSON.stringify(hfCard), /hf_[A-Za-z0-9]|sk-/);

  const parsed = parseLigaTrainingResult(`
LIGA_PROVIDER=aws-sagemaker
LIGA_S3_MODEL_ARTIFACT=s3://bucket/model.tar.gz
LIGA_EVAL_RESULT_JSON={"authorization":"Bearer ${'g'.repeat(32)}","eval_loss":0.1}
`);
  assert.equal(parsed?.s3ModelArtifact, 's3://bucket/model.tar.gz');
  assert.deepEqual(parsed?.evalResult, { authorization: 'Bearer [REDACTED]', eval_loss: 0.1 });
});
