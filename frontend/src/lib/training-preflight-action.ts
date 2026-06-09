import type { RunTrainingPreflightRequest } from './training-preflight-api.js';
import type { TrainingPreflightResult } from '../types/agent.js';
import { createTrainingPreflightPanel } from './training-preflight-panel.js';
import { redactJsonLike, redactText } from './redaction.js';

type PlannerRecord = Record<string, unknown>;

export type ManualPreflightStatus = 'not_run' | 'checking' | 'success' | 'error';

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

function safeRecord(value: PlannerRecord | null): Record<string, unknown> | undefined {
  if (!value || !Object.keys(value).length) return undefined;
  return redactJsonLike(value);
}

export function buildManualPreflightRequest(input: ManualPreflightRequestInput): RunTrainingPreflightRequest {
  const record = firstRecord(input.plannerOutput, input.plannerInput);
  const recommendation = getRecord(record, 'recommendation');
  const datasetSummary = getRecord(record, 'datasetSummary', 'dataset_summary');

  return {
    session_id: input.sessionId,
    ...(input.runId ? { run_id: input.runId } : {}),
    ...(safeRecord(recommendation) ? { recommendation: safeRecord(recommendation) } : {}),
    ...(safeRecord(datasetSummary) ? { dataset_summary: safeRecord(datasetSummary) } : {}),
    include_fallbacks: true,
    force_refresh: false,
    timeout_seconds: 15,
  };
}

export function createManualPreflightNotRunMarkdown(): string {
  return [
    '## Manual Preflight Check',
    '',
    `- ${PREFLIGHT_ACTION_COPY.staticNotVerified}`,
    `- ${PREFLIGHT_ACTION_COPY.notLaunch}`,
    `- ${PREFLIGHT_ACTION_COPY.noJobs}`,
    `- ${PREFLIGHT_ACTION_COPY.noResources}`,
    `- ${PREFLIGHT_ACTION_COPY.unknownNotPassed}`,
    `- ${PREFLIGHT_ACTION_COPY.approvalRequired}`,
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
