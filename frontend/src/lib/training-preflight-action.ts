import type { RunTrainingPreflightRequest } from './training-preflight-api.js';
import type { TrainingPreflightResult } from '../types/agent.js';
import { createTrainingPreflightPanel } from './training-preflight-panel.js';
import { redactJsonLike, redactText } from './redaction.js';

type PlannerRecord = Record<string, unknown>;

export type ManualPreflightStatus = 'not_run' | 'loading' | 'checking' | 'success' | 'error';

export interface ManualPreflightState {
  status: ManualPreflightStatus;
  disabled?: boolean;
  result?: TrainingPreflightResult;
  error?: string;
  markdown?: string;
  lastUpdated?: string;
}

export interface ManualPreflightRequestInput {
  sessionId: string;
  runId?: string | null;
  plannerOutput?: unknown;
  plannerInput?: unknown;
  forceRefresh?: boolean;
}

export const PREFLIGHT_ACTION_COPY = {
  button: 'Run preflight check',
  notLaunch: 'This is a preflight check, not a training launch.',
  noJobs: 'No provider jobs will be launched.',
  noResources: 'No resources will be created.',
  staticNotVerified: 'Static recommendation is not the same as verified launch readiness.',
  unknownNotPassed: 'Unknown does not mean passed.',
  approvalRequired: 'Launch still requires explicit approval.',
} as const;

function isRecord(value: unknown): value is PlannerRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function parseStructuredMarkdown(markdown: string): PlannerRecord | null {
  const structuredIndex = markdown.indexOf('### Structured Result');
  const searchArea = structuredIndex >= 0 ? markdown.slice(structuredIndex) : markdown;
  const jsonMatch = searchArea.match(/```json\s*([\s\S]*?)```/);
  if (!jsonMatch) return null;
  try {
    const parsed = JSON.parse(jsonMatch[1]);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function plannerRecord(value: unknown): PlannerRecord {
  if (isRecord(value)) return value;
  if (typeof value === 'string') return parseStructuredMarkdown(value) ?? {};
  return {};
}

function firstRecord(...values: unknown[]): PlannerRecord {
  for (const value of values) {
    const record = plannerRecord(value);
    if (Object.keys(record).length) return record;
  }
  return {};
}

function getRecord(record: PlannerRecord, ...keys: string[]): PlannerRecord | null {
  for (const key of keys) {
    const value = record[key];
    if (isRecord(value)) return value;
  }
  return null;
}

function knownValue(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.trim().length > 0 &&
    !['unknown', 'null', 'none'].includes(value.trim().toLowerCase())
  );
}

function recommendationBody(record: PlannerRecord): PlannerRecord {
  return getRecord(record, 'recommendation') ?? record;
}

function recordValue(record: PlannerRecord, ...keys: string[]): string | null {
  for (const key of keys) {
    const value = record[key];
    if (knownValue(value)) return value;
  }
  return null;
}

function isCompletePreflightRecommendation(value: PlannerRecord | null): value is PlannerRecord {
  if (!value || !Object.keys(value).length) return false;
  const body = recommendationBody(value);
  const provider = recordValue(value, 'provider', 'provider_id', 'cloud_provider')
    ?? recordValue(getRecord(body, 'selected_provider') ?? {}, 'provider_id');
  const modelId = recordValue(value, 'recommended_model', 'model_id')
    ?? recordValue(getRecord(body, 'selected_model') ?? {}, 'model_id');
  const hardwareId = recordValue(value, 'hardware_id')
    ?? recordValue(getRecord(body, 'selected_hardware') ?? {}, 'hardware_id');
  const outputPolicy = recordValue(value, 'output_policy')
    ?? recordValue(body, 'output_policy');

  return [provider, modelId, hardwareId, outputPolicy].every(knownValue);
}

function preflightRecommendation(record: PlannerRecord): PlannerRecord | null {
  const nested = getRecord(record, 'recommendation');
  const hasTopLevelRecommendationFields = [
    recordValue(record, 'provider', 'provider_id', 'cloud_provider'),
    recordValue(record, 'recommended_model', 'model_id'),
    recordValue(record, 'hardware_id'),
    recordValue(record, 'output_policy'),
  ].every(knownValue);
  if (nested && hasTopLevelRecommendationFields && isCompletePreflightRecommendation(record)) return record;
  if (isCompletePreflightRecommendation(nested)) return nested;
  if (isCompletePreflightRecommendation(record)) return record;
  return null;
}

function safeRecord(value: PlannerRecord | null): Record<string, unknown> | undefined {
  if (!value || !Object.keys(value).length) return undefined;
  return redactJsonLike(value);
}

export function buildManualPreflightRequest(input: ManualPreflightRequestInput): RunTrainingPreflightRequest {
  const record = firstRecord(input.plannerOutput, input.plannerInput);
  const recommendation = preflightRecommendation(record);
  const datasetSummary = getRecord(record, 'datasetSummary', 'dataset_summary');
  const safeRecommendation = safeRecord(recommendation);
  const safeDatasetSummary = safeRecord(datasetSummary);

  return {
    session_id: input.sessionId,
    ...(input.runId ? { run_id: input.runId } : {}),
    ...(safeRecommendation ? { recommendation: safeRecommendation } : {}),
    ...(safeDatasetSummary ? { dataset_summary: safeDatasetSummary } : {}),
    include_fallbacks: true,
    force_refresh: input.forceRefresh === true,
    timeout_seconds: 15,
  };
}

export function createManualPreflightNotRunMarkdown(): string {
  return [
    '## Manual Preflight Check',
    '',
    '- No preflight has been run yet.',
    `- ${PREFLIGHT_ACTION_COPY.staticNotVerified}`,
    `- ${PREFLIGHT_ACTION_COPY.notLaunch}`,
    `- ${PREFLIGHT_ACTION_COPY.noJobs}`,
    `- ${PREFLIGHT_ACTION_COPY.noResources}`,
    `- ${PREFLIGHT_ACTION_COPY.unknownNotPassed}`,
    `- ${PREFLIGHT_ACTION_COPY.approvalRequired}`,
  ].join('\n');
}

export function createPersistedPreflightLoadingMarkdown(): string {
  return [
    '## Latest Preflight',
    '',
    '- Loading latest persisted preflight result.',
    '- This is a read-only lookup. It does not run preflight.',
  ].join('\n');
}

export function createManualPreflightCheckingMarkdown(): string {
  return [
    '## Manual Preflight Check',
    '',
    '- Preflight is running.',
    `- ${PREFLIGHT_ACTION_COPY.notLaunch}`,
    `- ${PREFLIGHT_ACTION_COPY.noJobs}`,
    `- ${PREFLIGHT_ACTION_COPY.noResources}`,
  ].join('\n');
}

export function createManualPreflightErrorMarkdown(error: string): string {
  return [
    '## Manual Preflight Check',
    '',
    `- Error: ${redactText(error)}`,
    '- You can retry with Run preflight check.',
    `- ${PREFLIGHT_ACTION_COPY.notLaunch}`,
    `- ${PREFLIGHT_ACTION_COPY.noJobs}`,
    `- ${PREFLIGHT_ACTION_COPY.noResources}`,
  ].join('\n');
}

export function createPersistedPreflightErrorMarkdown(error: string): string {
  return [
    '## Latest Preflight',
    '',
    `- Error: ${redactText(error)}`,
    '- Static recommendation remains visible.',
    '- You can retry with Run preflight check or Refresh preflight.',
    `- ${PREFLIGHT_ACTION_COPY.notLaunch}`,
    `- ${PREFLIGHT_ACTION_COPY.noJobs}`,
    `- ${PREFLIGHT_ACTION_COPY.noResources}`,
  ].join('\n');
}

export function createPersistedPreflightMarkdown(result: TrainingPreflightResult): string {
  const panel = createTrainingPreflightPanel(result);
  return [
    panel.markdown,
    '',
    '### Persisted Result',
    '- Stored preflight may be stale. Refresh manually before launch.',
    `- Created at: ${redactText(result.created_at)}`,
    `- Updated at: ${redactText(result.updated_at)}`,
    `- ${PREFLIGHT_ACTION_COPY.approvalRequired}`,
  ].join('\n');
}

export async function loadPersistedTrainingPreflight({
  sessionId,
  runId,
  getLatest,
  getRun,
  onStateChange,
}: {
  sessionId: string;
  runId?: string | null;
  getLatest: (sessionId: string) => Promise<TrainingPreflightResult | null>;
  getRun: (sessionId: string, runId: string) => Promise<TrainingPreflightResult | null>;
  onStateChange: (state: ManualPreflightState) => void;
}): Promise<void> {
  onStateChange({
    status: 'loading',
    disabled: true,
    markdown: createPersistedPreflightLoadingMarkdown(),
  });

  try {
    const result = runId ? await getRun(sessionId, runId) : await getLatest(sessionId);
    if (!result) {
      onStateChange({
        status: 'not_run',
        disabled: false,
        markdown: createManualPreflightNotRunMarkdown(),
        lastUpdated: new Date().toISOString(),
      });
      return;
    }

    onStateChange({
      status: 'success',
      disabled: false,
      result,
      markdown: createPersistedPreflightMarkdown(result),
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    const message = redactText(error instanceof Error ? error.message : String(error));
    onStateChange({
      status: 'error',
      disabled: false,
      error: message,
      markdown: createPersistedPreflightErrorMarkdown(message),
      lastUpdated: new Date().toISOString(),
    });
  }
}

export async function runManualTrainingPreflight({
  request,
  run,
  onStateChange,
}: {
  request: RunTrainingPreflightRequest;
  run: (request: RunTrainingPreflightRequest) => Promise<TrainingPreflightResult>;
  onStateChange: (state: ManualPreflightState) => void;
}): Promise<void> {
  onStateChange({
    status: 'checking',
    disabled: true,
    markdown: createManualPreflightCheckingMarkdown(),
  });

  try {
    const result = await run(request);
    const panel = createTrainingPreflightPanel(result);
    onStateChange({
      status: 'success',
      disabled: false,
      result,
      markdown: panel.markdown,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    const message = redactText(error instanceof Error ? error.message : String(error));
    onStateChange({
      status: 'error',
      disabled: false,
      error: message,
      markdown: createManualPreflightErrorMarkdown(message),
      lastUpdated: new Date().toISOString(),
    });
  }
}
