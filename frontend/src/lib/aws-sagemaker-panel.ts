import { parseLigaTrainingResult, type TrainingResult } from '../utils/trainingResult.js';

interface PanelData {
  title: string;
  script?: { content: string; language: string };
  output?: { content: string; language: string };
  input?: { content: string; language: string };
  parameters?: Record<string, unknown>;
}

export interface AwsSageMakerToolState {
  state?: string;
  jobName?: string;
  jobUrl?: string;
  region?: string;
  s3TrainUri?: string;
  s3OutputUri?: string;
  s3ModelArtifact?: string;
  cloudWatchLogsUrl?: string;
  outputPolicy?: string;
}

const AWS_SUMMARY_FIELDS = [
  ['Template', 'template'],
  ['Dataset', 'dataset_name'],
  ['Dataset config', 'dataset_config'],
  ['Dataset split', 'dataset_split'],
  ['Model', 'model_name'],
  ['Output model ID', 'output_model_id'],
  ['Output policy', 'output_policy'],
  ['Instance type', 'instance_type'],
  ['Instance count', 'instance_count'],
  ['Max runtime seconds', 'max_run_seconds'],
  ['S3 bucket', 's3_bucket'],
  ['S3 prefix', 's3_prefix'],
] as const;

const AWS_RESULT_HEADING = '## Liga Training Result';

function valueToString(value: unknown): string | null {
  if (value === undefined || value === null || value === '') return null;
  if (Array.isArray(value)) return value.map(String).join(' ');
  return String(value);
}

function renderValue(label: string, value: unknown): string | null {
  const text = valueToString(value);
  if (!text) return null;
  const isLink = label.includes('URL') || label.includes('console') || label.includes('logs') || label === 'Final model';
  const rendered = isLink && text.startsWith('http') ? `[${text}](${text})` : `\`${text}\``;
  return `| ${label} | ${rendered} |`;
}

export function buildAwsSftSummary(args: Record<string, unknown>): string {
  const rows = AWS_SUMMARY_FIELDS
    .map(([label, key]) => renderValue(label, args[key]))
    .filter((row): row is string => Boolean(row));

  return [
    '## AWS SageMaker Training Job',
    '',
    '| Field | Value |',
    '| --- | --- |',
    ...rows,
  ].join('\n');
}

export function buildAwsStateMarkdown(state: AwsSageMakerToolState): string {
  const rows = [
    ['Status', state.state],
    ['Job name', state.jobName],
    ['Region', state.region],
    ['S3 train URI', state.s3TrainUri],
    ['S3 output URI', state.s3OutputUri],
    ['S3 model artifact', state.s3ModelArtifact],
    ['CloudWatch logs URL', state.cloudWatchLogsUrl],
    ['SageMaker console URL', state.jobUrl],
    ['Output policy', state.outputPolicy],
  ]
    .map(([label, value]) => renderValue(String(label), value))
    .filter((row): row is string => Boolean(row));

  if (rows.length === 0) return '';

  return [
    '## AWS SageMaker Job State',
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

export function buildAwsTrainingResultMarkdown(result: TrainingResult): string {
  const rows = [
    ['Status', result.status],
    ['Provider', result.provider],
    ['SageMaker job name', result.awsTrainingJobName],
    ['Region', result.awsRegion],
    ['S3 model artifact', result.s3ModelArtifact],
    ['S3 output dir', result.s3OutputDir],
    ['CloudWatch logs URL', result.cloudWatchLogsUrl],
    ['Final model', result.finalModelUrl],
    ['Hub model ID', result.hubModelId],
    ['Result file', result.resultFile],
  ]
    .map(([label, value]) => {
      const text = resultValue(value);
      if (!text) return null;
      return renderValue(String(label), text);
    })
    .filter((row): row is string => Boolean(row));

  const sections = [
    AWS_RESULT_HEADING,
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

export function appendAwsTrainingResultSummary(output: string): string {
  const result = parseLigaTrainingResult(output);
  const withoutPreviousSummary = output
    .replace(new RegExp(`\\n*${AWS_RESULT_HEADING}[\\s\\S]*$`), '')
    .trimEnd();

  if (!result) return withoutPreviousSummary || output;
  if (result.provider !== 'aws-sagemaker') return withoutPreviousSummary || output;
  return [withoutPreviousSummary, buildAwsTrainingResultMarkdown(result)]
    .filter(Boolean)
    .join('\n\n');
}

export function createAwsSageMakerRunPanel(args: Record<string, unknown>): {
  data: PanelData;
  view: 'script' | 'output';
  editable: boolean;
} | null {
  if (args.operation !== 'run') return null;

  if (args.template === 'sft' || !args.template) {
    return {
      data: {
        title: 'AWS SageMaker SFT Training',
        output: { content: buildAwsSftSummary(args), language: 'markdown' },
        parameters: args,
      },
      view: 'output',
      editable: false,
    };
  }

  return null;
}
