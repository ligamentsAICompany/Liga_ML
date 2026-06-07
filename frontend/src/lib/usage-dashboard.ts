import type { UsageEntry, UsageProviderId, UsageSummary } from '../types/usage.js';
import { redactJsonLike, redactText } from './redaction.js';

export const PROVIDER_LABELS: Record<UsageProviderId, string> = {
  'hf-jobs': 'HF Jobs',
  'gcp-vertex': 'Google Cloud Vertex AI',
  'aws-sagemaker': 'AWS SageMaker AI',
  llm: 'LLM / Agent model',
  unknown: 'Unknown provider',
};

export interface ProviderUsageCard {
  provider: UsageProviderId;
  label: string;
  configured: boolean;
  ready: boolean;
  estimatedCostUsd: number;
  knownCostUsd: number;
  recentJobs: UsageEntry[];
  warnings: string[];
}

export function formatUsd(value: number | null | undefined): string {
  if (value === null || value === undefined) return 'Unknown';
  return `$${value.toFixed(2)}`;
}

export function providerLabel(provider: string | null | undefined): string {
  return PROVIDER_LABELS[(provider || 'unknown') as UsageProviderId] || String(provider || 'Unknown provider');
}

function readinessFor(summary: UsageSummary, provider: UsageProviderId): { configured: boolean; ready: boolean; warnings: string[] } {
  const readinessKey = provider === 'hf-jobs' ? 'hf_jobs' : provider === 'gcp-vertex' ? 'gcp_vertex' : provider === 'aws-sagemaker' ? 'aws_sagemaker' : provider;
  const raw = summary.provider_readiness?.[readinessKey] as Record<string, unknown> | undefined;
  const warnings = [
    ...((raw?.notes as string[] | undefined) || []),
    ...((raw?.warnings as string[] | undefined) || []),
    ...((raw?.errors as string[] | undefined) || []),
  ];
  return {
    configured: Boolean(raw?.configured),
    ready: Boolean(raw?.configured) && !warnings.some((message) => /missing|required|quota|error/i.test(message)),
    warnings,
  };
}

export function buildProviderCards(summary: UsageSummary): ProviderUsageCard[] {
  const safeSummary = redactJsonLike(summary);
  const entries = safeSummary.recent_usage_entries || [];
  return (['hf-jobs', 'gcp-vertex', 'aws-sagemaker', 'llm'] as UsageProviderId[]).map((provider) => {
    const cost = safeSummary.cost_by_provider?.[provider];
    const readiness = readinessFor(safeSummary, provider);
    const providerWarnings = [
      ...readiness.warnings,
      ...safeSummary.budget_warnings.filter((item) => item.provider === provider).map((item) => item.message || ''),
      ...safeSummary.quota_warnings.filter((item) => item.provider === provider).map((item) => item.message || ''),
    ].filter(Boolean);
    return {
      provider,
      label: providerLabel(provider),
      configured: readiness.configured,
      ready: readiness.ready,
      estimatedCostUsd: cost?.estimated_cost_usd ?? 0,
      knownCostUsd: cost?.known_cost_usd ?? 0,
      recentJobs: entries.filter((entry) => entry.provider === provider).slice(0, 5),
      warnings: Array.from(new Set(providerWarnings)),
    };
  });
}

export function usageEntryTitle(entry: UsageEntry): string {
  return redactText(entry.job_id || entry.run_id || entry.approval_id || entry.usage_id);
}
