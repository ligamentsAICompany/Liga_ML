import type {
  TrainingPreflightCheck,
  TrainingPreflightFallbackResult,
  TrainingPreflightResult,
  TrainingPreflightStatus,
} from '../types/agent.js';
import { redactJsonLike, redactText } from './redaction.js';

type PreflightRecord = Record<string, unknown>;

export interface TrainingPreflightPanel {
  title: string;
  markdown: string;
  status: string;
  launchReady: boolean;
}

const STATUS_DESCRIPTIONS: Record<string, string> = {
  not_run: 'Preflight not run',
  checking: 'Preflight is checking',
  passed: 'Passed',
  warning: 'Ready with warnings or review required',
  failed: 'Not launch-ready',
  unknown: 'Not proven / not launch-ready by default',
  skipped: 'Skipped / not applicable',
};

function isRecord(value: unknown): value is PreflightRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function cleanText(value: unknown): string | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string') return redactText(value);
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

function cleanJsonText(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  return redactText(JSON.stringify(redactJsonLike(value)));
}

function line(label: string, value: unknown): string | null {
  const text = cleanText(value);
  return text ? `${label}: ${text}` : null;
}

function appendSection(lines: string[], title: string, items: Array<string | null>): void {
  const cleanItems = items.filter((item): item is string => !!item);
  lines.push('', `### ${title}`);
  if (!cleanItems.length) {
    lines.push('- None reported');
    return;
  }
  lines.push(...cleanItems.map((item) => `- ${item}`));
}

function statusDescription(status: string, launchReady = false): string {
  if (status === 'passed' && launchReady) return 'Passed and launch-ready';
  return STATUS_DESCRIPTIONS[status] ?? cleanText(status) ?? 'Unknown status';
}

function checkLine(check: TrainingPreflightCheck): string {
  const status = cleanText(check.status) ?? 'unknown';
  const description = statusDescription(status, false);
  const parts = [
    cleanText(check.label) ?? cleanText(check.check_id) ?? 'Unnamed check',
    cleanText(check.category) ? `category: ${cleanText(check.category)}` : null,
    `status: ${status}`,
    description,
    cleanText(check.severity) ? `severity: ${cleanText(check.severity)}` : null,
    cleanText(check.message),
    cleanText(check.error_code) ? `error: ${cleanText(check.error_code)}` : null,
  ].filter((item): item is string => !!item);
  return parts.join(' | ');
}

function recommendationRecord(result: TrainingPreflightResult): PreflightRecord {
  const verified = result.verified_recommendation;
  if (!isRecord(verified)) return {};
  const nested = verified.recommendation;
  return isRecord(nested) ? nested : verified;
}

function recommendationLines(result: TrainingPreflightResult): string[] {
  const recommendation = recommendationRecord(result);
  const provider = isRecord(recommendation.selected_provider) ? recommendation.selected_provider : {};
  const model = isRecord(recommendation.selected_model) ? recommendation.selected_model : {};
  const hardware = isRecord(recommendation.selected_hardware) ? recommendation.selected_hardware : {};
  return [
    line('Provider', provider.display_name ?? provider.provider_id ?? result.provider),
    line('Provider ID', provider.provider_id ?? result.provider),
    line('Model', model.model_id ?? result.model_id),
    line('Hardware', hardware.hardware_id ?? hardware.display_name ?? result.hardware_id),
    line('Output policy', result.output_policy),
    'Static recommendation and preflight verification are different.',
    'Static recommendation is not launch-ready by itself.',
  ].filter((item): item is string => !!item);
}

function fallbackLine(fallback: TrainingPreflightFallbackResult): string {
  const parts = [
    cleanText(fallback.fallback_id) ?? 'fallback',
    cleanText(fallback.provider) ? `provider: ${cleanText(fallback.provider)}` : null,
    cleanText(fallback.model_id) ? `model: ${cleanText(fallback.model_id)}` : null,
    cleanText(fallback.hardware_id) ? `hardware: ${cleanText(fallback.hardware_id)}` : null,
    `status: ${cleanText(fallback.status) ?? 'unknown'}`,
    `launch ready: ${fallback.launch_ready ? 'yes' : 'no'}`,
    cleanText(fallback.reason),
  ].filter((item): item is string => !!item);
  return parts.join(' | ');
}

function verifiedFallbackLines(result: TrainingPreflightResult): string[] {
  const fallback = result.verified_fallback;
  if (!fallback) {
    return ['No verified fallback is available from this preflight result.'];
  }
  return [
    fallbackLine(fallback),
    'Verified fallback is advisory only.',
    'Fallback was not automatically launched.',
    'Fallback launch still requires explicit approval.',
    'No jobs or resources were created for the fallback.',
  ];
}

function metadataBool(metadata: TrainingPreflightResult['metadata'], key: 'provider_jobs_launched' | 'resources_created'): boolean {
  return (metadata as Record<string, unknown> | undefined)?.[key] === true;
}

function safetyLines(result: TrainingPreflightResult): string[] {
  const providerJobsLaunched = metadataBool(result.metadata, 'provider_jobs_launched');
  const resourcesCreated = metadataBool(result.metadata, 'resources_created');
  const manualAllowed = result.manual_approval_allowed === true;
  const lines = [
    'This result is a preflight check, not a training launch.',
    `No provider jobs were launched: ${String(!providerJobsLaunched)}`,
    `No resources were created: ${String(!resourcesCreated)}`,
    'Launch still requires explicit approval.',
    'Unknown does not mean passed.',
  ];
  if (manualAllowed) {
    lines.push(
      'Preflight has unknowns; bounded smoke can proceed only with explicit approval.',
    );
    if (result.manual_approval_reason) {
      lines.push(cleanText(result.manual_approval_reason) ?? result.manual_approval_reason);
    }
  }
  return lines;
}

function cacheLines(result: TrainingPreflightResult): string[] {
  const cache = result.cache ?? { hit: false };
  return [
    line('Cache hit', cache.hit),
    line('Cache key', cache.cache_key),
    line('TTL seconds', cache.ttl_seconds),
    line('Created at', cache.created_at),
    line('Expires at', cache.expires_at),
  ].filter((item): item is string => !!item);
}

function timestampLines(result: TrainingPreflightResult): string[] {
  return [
    line('Created at', result.created_at),
    line('Updated at', result.updated_at),
    line('Run ID', result.run_id),
  ].filter((item): item is string => !!item);
}

function parseInput(input: unknown): PreflightRecord {
  if (isRecord(input)) return input;
  if (typeof input !== 'string') return {};
  try {
    const parsed = JSON.parse(input);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function normalizeResult(input: unknown): TrainingPreflightResult {
  const record = parseInput(input);
  if (record.preflight_id && record.primary) return record as unknown as TrainingPreflightResult;
  return {
    preflight_id: cleanText(record.preflight_id) ?? 'not-run',
    session_id: cleanText(record.session_id) ?? 'unknown-session',
    run_id: cleanText(record.run_id),
    created_at: cleanText(record.created_at) ?? '',
    updated_at: cleanText(record.updated_at) ?? '',
    status: cleanText(record.status) ?? 'not_run',
    launch_ready: record.launch_ready === true,
    provider: cleanText(record.provider) ?? 'unknown',
    model_id: cleanText(record.model_id) ?? 'Not specified',
    hardware_id: cleanText(record.hardware_id),
    output_policy: cleanText(record.output_policy) ?? 'Not specified',
    primary: {
      provider: cleanText(record.provider) ?? 'unknown',
      status: cleanText(record.status) ?? 'not_run',
      launch_ready: record.launch_ready === true,
      checks: [],
      blocking_reasons: [],
      warning_reasons: [],
      unknown_reasons: [],
      metadata: {},
    },
    fallbacks: [],
    verified_fallback: null,
    verified_recommendation: null,
    blocking_reasons: [],
    warning_reasons: [],
    unknown_reasons: [],
    safe_summary: cleanText(record.safe_summary) ?? 'Preflight not run.',
    manual_approval_allowed: record.manual_approval_allowed === true,
    manual_approval_reason: cleanText(record.manual_approval_reason),
    approval_required: record.approval_required === true,
    cache: { hit: false },
    metadata: {
      provider_jobs_launched: false,
      resources_created: false,
      live_checks_optional: true,
    },
  };
}

export function createTrainingPreflightPanel(input: unknown): TrainingPreflightPanel {
  const result = normalizeResult(input);
  const status = cleanText(result.status) ?? 'unknown';
  const launchReady = result.launch_ready === true;
  const manualAllowed = result.manual_approval_allowed === true;
  const primaryChecks = result.primary?.checks ?? [];
  const summary = cleanText(result.safe_summary) ?? statusDescription(status, launchReady);

  const lines = [
    '## Live Preflight Summary',
    '',
    `- Status: ${status} (${statusDescription(status, launchReady)})`,
    `- Launch ready: ${launchReady ? 'yes' : 'no'}`,
    `- Summary: ${summary}`,
  ];

  appendSection(lines, 'Launch Readiness', [
    launchReady
      ? 'Launch ready: yes, pending explicit user approval.'
      : manualAllowed
        ? 'Launch ready: no. Only quota/accelerator or safe GCS write-readiness checks are unknown; bounded smoke may proceed with explicit approval.'
        : 'Launch ready: no. Local/static checks are not enough to prove provider readiness.',
    manualAllowed
      ? 'Preflight has unknowns; bounded smoke can proceed only with explicit approval.'
      : null,
    manualAllowed && result.manual_approval_reason
      ? cleanText(result.manual_approval_reason)
      : null,
    status === 'unknown' ? 'Unknown checks are not treated as passed.' : null,
    status === 'checking' ? 'Preflight is checking; do not treat this as final approval.' : null,
    status === 'not_run' ? 'Preflight not run for this session or run.' : null,
  ]);

  appendSection(lines, 'Primary Recommendation', recommendationLines(result));
  appendSection(lines, 'Result Timestamps', timestampLines(result));
  appendSection(lines, 'Checks', primaryChecks.map(checkLine));
  appendSection(lines, 'Blocking Reasons', (result.blocking_reasons ?? []).map(cleanText));
  appendSection(lines, 'Warnings', (result.warning_reasons ?? []).map(cleanText));
  appendSection(lines, 'Unknowns', (result.unknown_reasons ?? []).map(cleanText));
  appendSection(lines, 'Fallbacks', (result.fallbacks ?? []).map(fallbackLine));
  appendSection(lines, 'Verified Fallback', verifiedFallbackLines(result));
  appendSection(lines, 'Safety Metadata', safetyLines(result));
  appendSection(lines, 'Cache Info', cacheLines(result));

  const metadata = cleanJsonText(result.metadata);
  if (metadata) {
    appendSection(lines, 'Metadata', [`\`${metadata}\``]);
  }

  return {
    title: 'Training Preflight',
    markdown: lines.join('\n'),
    status,
    launchReady,
  };
}

export function trainingPreflightStatusLabel(status: TrainingPreflightStatus | string, launchReady = false): string {
  return statusDescription(String(status), launchReady);
}
