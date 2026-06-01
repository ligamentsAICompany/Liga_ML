import type { OutputPolicy } from '../types/agent.js';
import { outputPolicyLabel } from './output-policy.js';

type PlannerRecord = Record<string, unknown>;

export interface TrainingPlannerPanel {
  title: string;
  summaryLines: string[];
  warningLines: string[];
  riskLines: string[];
  reasoningLines: string[];
  nextStepText: string;
  markdown: string;
}

const FIELD_NAMES = {
  trainingGoal: ['trainingGoal', 'training_goal'],
  recommendedModel: ['recommendedModel', 'recommended_model'],
  smokeTestModel: ['smokeTestModel', 'smoke_test_model'],
  productionModel: ['productionModel', 'production_model'],
  recommendedHardware: ['recommendedHardware', 'recommended_hardware'],
  trainingArgs: ['trainingArgs', 'training_args'],
  outputPolicy: ['outputPolicy', 'output_policy'],
  privacyWarnings: ['privacyWarnings', 'privacy_warnings'],
  risks: ['risks'],
  reasoning: ['reasoning'],
  provider: ['provider'],
  domain: ['domain'],
  datasetSummary: ['datasetSummary', 'dataset_summary'],
} as const;

function isRecord(value: unknown): value is PlannerRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function humanizeKey(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase();
}

function valueLabel(value: unknown): string | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

function getValue(record: PlannerRecord, names: readonly string[]): unknown {
  for (const name of names) {
    if (record[name] !== undefined) return record[name];
  }
  return undefined;
}

function getString(record: PlannerRecord, names: readonly string[]): string | null {
  return valueLabel(getValue(record, names));
}

function getList(record: PlannerRecord, names: readonly string[]): string[] {
  const value = getValue(record, names);
  if (!Array.isArray(value)) return [];
  return value.map(valueLabel).filter((item): item is string => !!item);
}

function normalizePolicy(value: string | null): OutputPolicy | null {
  if (value === 'cloud-private' || value === 'hf-hub' || value === 'cloud-and-hf-hub') {
    return value;
  }
  return null;
}

function formatObject(value: unknown): string | null {
  if (!isRecord(value)) return valueLabel(value);
  const entries = Object.entries(value)
    .map(([key, item]) => {
      const label = valueLabel(item);
      return label ? `${humanizeKey(key)}: ${label}` : null;
    })
    .filter((item): item is string => !!item);
  return entries.length ? entries.join(', ') : null;
}

function formatDatasetSummary(value: unknown): string | null {
  if (!isRecord(value)) return null;
  const rows = valueLabel(value.rows);
  const sourceFormat = valueLabel(value.source_format ?? value.sourceFormat);
  const columns = Array.isArray(value.columns)
    ? value.columns.map(valueLabel).filter(Boolean).join(', ')
    : null;
  const parts = [
    rows ? `${Number(rows).toLocaleString()} rows` : null,
    sourceFormat ? sourceFormat.toUpperCase() : null,
    columns ? `columns: ${columns}` : null,
  ].filter((item): item is string => !!item);
  return parts.length ? parts.join(' · ') : null;
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

function extractPlannerRecord(input: unknown): { record: PlannerRecord; fallbackMarkdown: string | null } {
  if (isRecord(input)) return { record: input, fallbackMarkdown: null };
  if (typeof input === 'string') {
    return {
      record: parseStructuredMarkdown(input) ?? {},
      fallbackMarkdown: input,
    };
  }
  return { record: {}, fallbackMarkdown: null };
}

function appendSection(lines: string[], title: string, items: string[]): void {
  if (!items.length) return;
  lines.push('', `### ${title}`, ...items.map((item) => `- ${item}`));
}

export function createTrainingPlannerPanel(input: unknown): TrainingPlannerPanel {
  const { record, fallbackMarkdown } = extractPlannerRecord(input);
  const provider = getString(record, FIELD_NAMES.provider) ?? 'hf-jobs';
  const outputPolicy = normalizePolicy(getString(record, FIELD_NAMES.outputPolicy));
  const hardware = formatObject(getValue(record, FIELD_NAMES.recommendedHardware));
  const trainingArgs = formatObject(getValue(record, FIELD_NAMES.trainingArgs));
  const datasetSummary = formatDatasetSummary(getValue(record, FIELD_NAMES.datasetSummary));
  const privacyWarnings = getList(record, FIELD_NAMES.privacyWarnings);
  const risks = getList(record, FIELD_NAMES.risks);
  const reasoning = getList(record, FIELD_NAMES.reasoning);

  const summaryLines = [
    `Provider: ${provider}`,
    `Goal: ${getString(record, FIELD_NAMES.trainingGoal) ?? 'Not specified yet'}`,
    `Recommended model: ${getString(record, FIELD_NAMES.recommendedModel) ?? 'Not specified yet'}`,
    `Smoke-test model: ${getString(record, FIELD_NAMES.smokeTestModel) ?? 'Not specified yet'}`,
    `Production model: ${getString(record, FIELD_NAMES.productionModel) ?? 'Not specified yet'}`,
    `Hardware: ${hardware ?? 'Not specified yet'}`,
    `Output policy: ${outputPolicy ? outputPolicyLabel(provider, outputPolicy) : 'Not specified yet'}`,
  ];
  if (getString(record, FIELD_NAMES.domain)) {
    summaryLines.splice(1, 0, `Domain: ${getString(record, FIELD_NAMES.domain)}`);
  }
  if (datasetSummary) {
    summaryLines.push(`Dataset: ${datasetSummary}`);
  }
  if (trainingArgs) {
    summaryLines.push(`Training args: ${trainingArgs}`);
  }

  const missingDataset = risks.some((risk) => /no training dataset|dataset discovery|required before final/i.test(risk));
  const nextStepText = missingDataset
    ? 'No uploaded dataset is attached. Dataset discovery is required before training. User approval is required before any billable cloud job.'
    : 'Planning only: the planner recommends settings; it does not launch training. User approval is required before any billable cloud job.';

  const warningLines = [...privacyWarnings];
  const riskLines = [...risks];

  if (!Object.keys(record).length && fallbackMarkdown) {
    return {
      title: 'Training Planner Recommendation',
      summaryLines: ['Structured planner fields were not available; showing the planner output as markdown.'],
      warningLines: [],
      riskLines: [],
      reasoningLines: [],
      nextStepText,
      markdown: fallbackMarkdown,
    };
  }

  const lines = [
    '## Training Planner Recommendation',
    '',
    ...summaryLines.map((line) => `- ${line}`),
  ];
  appendSection(lines, 'Privacy Warnings', warningLines);
  appendSection(lines, 'Risks', riskLines);
  appendSection(lines, 'Reasoning', reasoning);
  lines.push('', nextStepText);

  return {
    title: 'Training Planner Recommendation',
    summaryLines,
    warningLines,
    riskLines,
    reasoningLines: reasoning,
    nextStepText,
    markdown: lines.join('\n'),
  };
}
