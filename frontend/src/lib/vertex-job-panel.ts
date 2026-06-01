import type { PanelData } from '@/store/agentStore';
import { parseLigaTrainingResult, type TrainingResult } from '@/utils/trainingResult';
import { outputPolicyLabel, storageDestinationLabel, trainingGoalLabel } from '@/lib/gcloud-preflight';
import type { OutputPolicy, TrainingGoal } from '@/types/agent';

export interface VertexToolState {
  state?: string;
  jobName?: string;
  jobUrl?: string;
  outputDir?: string;
}

const VERTEX_SUMMARY_FIELDS = [
  ['Dataset', 'dataset_name'],
  ['Dataset config', 'dataset_config'],
  ['Dataset split', 'dataset_split'],
  ['Model', 'model_name'],
  ['Training goal', 'training_goal'],
  ['Output policy', 'output_policy'],
  ['HF target', 'hub_model_id'],
  ['Machine type', 'machine_type'],
  ['Accelerator type', 'accelerator_type'],
  ['Accelerator count', 'accelerator_count'],
  ['Output dir', 'output_dir'],
  ['Staging bucket', 'staging_bucket'],
  ['Trackio project', 'trackio_project'],
  ['Trackio Space', 'trackio_space_id'],
] as const;
const TRAINING_RESULT_HEADING = '## Liga Training Result';
const SECRET_KEY_PATTERN = /token|secret|password|credential|private_key/i;

function valueToString(value: unknown): string | null {
  if (value === undefined || value === null || value === '') return null;
  if (Array.isArray(value)) return value.map(String).join(' ');
  return String(value);
}

function maskSensitiveParameters(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(maskSensitiveParameters);
  if (!value || typeof value !== 'object') return value;
  const masked: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (SECRET_KEY_PATTERN.test(key)) {
      masked[key] = '[REDACTED]';
    } else {
      masked[key] = maskSensitiveParameters(entry);
    }
  }
  return masked;
}

function summaryValue(label: string, value: unknown): string | null {
  if (label === 'Training goal') {
    return trainingGoalLabel(value as TrainingGoal | undefined);
  }
  if (label === 'Output policy') {
    return outputPolicyLabel(value as OutputPolicy | undefined);
  }
  return valueToString(value);
}

export function buildVertexSftSummary(args: Record<string, unknown>): string {
  const rows = VERTEX_SUMMARY_FIELDS
    .map(([label, key]) => {
      const value = summaryValue(label, args[key]);
      return value ? `| ${label} | \`${value}\` |` : null;
    })
    .filter((row): row is string => Boolean(row));

  return [
    '## Vertex AI SFT Training',
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ].join('\n');
}

export function buildVertexStateMarkdown(state: VertexToolState): string {
  const rows = [
    ['State', state.state],
    ['Vertex job', state.jobName],
    ['GCS output directory', state.outputDir],
    ['Vertex console', state.jobUrl],
  ]
    .map(([label, value]) => {
      const text = valueToString(value);
      if (!text) return null;
      const rendered = label === 'Vertex console' ? `[${text}](${text})` : `\`${text}\``;
      return `| ${label} | ${rendered} |`;
    })
    .filter((row): row is string => Boolean(row));

  if (rows.length === 0) return '';

  return [
    '## Vertex AI Job State',
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ].join('\n');
}

function resultValue(value: unknown): string | null {
  if (value === undefined || value === null || value === '') return null;
  return String(value);
}

export function buildTrainingResultMarkdown(result: TrainingResult): string {
  const rows = [
    ['Status', result.status],
    ['Provider', result.provider],
    ['Output policy', result.outputPolicy],
    ['Final HF model', result.finalModelUrl],
    ['Hub model ID', result.hubModelId],
    ['GCS output directory', result.gcsOutputDir],
    ['Result file', result.resultFile],
  ]
    .map(([label, value]) => {
      const text = resultValue(value);
      if (!text) return null;
      const rendered = label === 'Final HF model' && text.startsWith('https://')
        ? `[${text}](${text})`
        : `\`${text}\``;
      return `| ${label} | ${rendered} |`;
    })
    .filter((row): row is string => Boolean(row));

  const sections = [
    TRAINING_RESULT_HEADING,
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ];

  if (result.evalResult !== undefined) {
    if (
      result.evalResult &&
      typeof result.evalResult === 'object' &&
      Object.keys(result.evalResult).length === 0
    ) {
      sections.push('', '**Evaluation result:** No evaluation metrics were reported; evaluation was skipped or empty.');
    } else {
      sections.push(
        '',
        '**Evaluation result:**',
        '',
        '```json',
        JSON.stringify(result.evalResult, null, 2),
        '```',
      );
    }
  }

  return sections.join('\n');
}

export function appendTrainingResultSummary(output: string): string {
  const result = parseLigaTrainingResult(output);
  const withoutPreviousSummary = output
    .replace(new RegExp(`\\n*${TRAINING_RESULT_HEADING}[\\s\\S]*$`), '')
    .trimEnd();

  if (!result) return withoutPreviousSummary || output;
  return [withoutPreviousSummary, buildTrainingResultMarkdown(result)]
    .filter(Boolean)
    .join('\n\n');
}

export function createVertexRunPanel(args: Record<string, unknown>): {
  data: PanelData;
  view: 'script' | 'output';
  editable: boolean;
} | null {
  if (args.operation !== 'run') return null;

  if (typeof args.script === 'string' && args.script) {
    return {
      data: {
        title: 'Vertex AI Script',
        script: { content: args.script, language: 'python' },
        parameters: maskSensitiveParameters(args) as Record<string, unknown>,
      },
      view: 'script',
      editable: false,
    };
  }

  if (args.template === 'sft') {
    const policy = args.output_policy as OutputPolicy | undefined;
    return {
      data: {
        title: 'Vertex AI SFT Training',
        output: {
          content: [
            buildVertexSftSummary(args),
            `**Storage destination:** ${storageDestinationLabel(policy)}`,
          ].join('\n\n'),
          language: 'markdown',
        },
        parameters: maskSensitiveParameters(args) as Record<string, unknown>,
      },
      view: 'output',
      editable: false,
    };
  }

  if (Array.isArray(args.command) && args.command.length > 0) {
    return {
      data: {
        title: 'Vertex AI Command',
        script: { content: args.command.map(String).join(' '), language: 'bash' },
        parameters: maskSensitiveParameters(args) as Record<string, unknown>,
      },
      view: 'script',
      editable: false,
    };
  }

  return null;
}
