import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  buildProviderCards,
  formatUsd,
  providerLabel,
  usageEntryTitle,
} from '../src/lib/usage-dashboard.js';
import type { UsageSummary } from '../src/types/usage.js';

const summary: UsageSummary = {
  total_estimated_cost_usd: 2.5,
  total_known_cost_usd: 0,
  cost_by_provider: {
    'hf-jobs': { estimated_cost_usd: 1, known_cost_usd: 0, count: 1 },
    'aws-sagemaker': { estimated_cost_usd: 1.5, known_cost_usd: 0, count: 1 },
  },
  cost_by_session: {},
  cost_by_run: {},
  recent_usage_entries: [
    {
      usage_id: 'u1',
      session_id: 's1',
      run_id: 'r1',
      provider: 'hf-jobs',
      tool_name: 'hf_jobs',
      operation: 'run',
      job_id: 'job-1',
      job_url: 'https://huggingface.co/jobs/job-1',
      artifact_url: 'https://huggingface.co/models/demo/model',
      status: 'approval_required',
      currency: 'USD',
      estimated_cost_usd: 1,
      known_cost_usd: null,
      cost_source: 'approval_estimate',
      cost_confidence: 'estimated',
      approved: false,
      quota_status: 'unknown',
      warning: 'Estimated cost, not final bill',
    },
  ],
  budget_warnings: [{ provider: 'hf-jobs', message: 'Estimated cost exceeds budget', usage_id: 'u1' }],
  quota_warnings: [{ provider: 'aws-sagemaker', message: 'ml.g5.xlarge training quota is 0', usage_id: 'u2' }],
  provider_readiness: {
    hf_jobs: { configured: true, notes: [] },
    gcp_vertex: { configured: false, warnings: ['Missing project'] },
    aws_sagemaker: { configured: true, errors: ['ml.g5.xlarge training quota is 0'] },
  },
  usage_store: { enabled: true, durable: true, store: 'mongodb' },
};

test('usage dashboard builds provider cards', () => {
  const cards = buildProviderCards(summary);

  assert.equal(cards.length, 4);
  assert.equal(cards[0].label, 'HF Jobs');
  assert.equal(cards[0].estimatedCostUsd, 1);
  assert.equal(cards[0].recentJobs[0].job_url, 'https://huggingface.co/jobs/job-1');
});

test('usage dashboard empty state has zero-cost provider cards', () => {
  const cards = buildProviderCards({ ...summary, recent_usage_entries: [], cost_by_provider: {} });

  assert.equal(cards[0].estimatedCostUsd, 0);
  assert.deepEqual(cards[0].recentJobs, []);
});

test('usage labels make estimates and providers clear', () => {
  assert.equal(formatUsd(1.234), '$1.23');
  assert.equal(formatUsd(null), 'Unknown');
  assert.equal(providerLabel('gcp-vertex'), 'Google Cloud Vertex AI');
});

test('usage warnings and recent job title are exposed', () => {
  const cards = buildProviderCards(summary);
  const hf = cards.find((card) => card.provider === 'hf-jobs');
  const aws = cards.find((card) => card.provider === 'aws-sagemaker');

  assert.ok(hf?.warnings.some((message) => message.includes('budget')));
  assert.ok(aws?.warnings.some((message) => message.includes('quota')));
  assert.equal(usageEntryTitle(summary.recent_usage_entries[0]), 'job-1');
});
