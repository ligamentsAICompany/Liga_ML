import { redactJsonLike, redactText } from './redaction.js';

export type EvaluationStatus =
  | 'not_started'
  | 'planned'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'skipped'
  | 'unavailable';

export interface EvaluationScores {
  overall_score?: number | null;
  task_relevance_score?: number | null;
  safety_score?: number | null;
  privacy_score?: number | null;
  metric_quality_score?: number | null;
  confidence?: number | null;
}

export interface PostTrainingEvaluation {
  evaluation_id: string;
  session_id: string;
  run_id: string;
  provider?: string;
  job_id?: string | null;
  model_ref?: string | null;
  artifact_ref?: string | null;
  dataset_ref?: string | null;
  status: EvaluationStatus | string;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  evaluation_type?: string;
  domain?: string;
  task_type?: string;
  test_prompts?: string[];
  results?: Record<string, unknown>;
  scores?: EvaluationScores;
  safety_findings?: Array<{ severity?: string; message?: string }>;
  privacy_findings?: Array<{ severity?: string; message?: string }>;
  quality_summary?: string | null;
  failure_summary?: string | null;
  recommendation?: string | null;
  report_markdown?: string | null;
  artifact_paths?: string[];
  metadata?: Record<string, unknown>;
}

export function evaluationStatusLabel(status?: string | null): string {
  const normalized = (status || 'not_started').toLowerCase();
  const labels: Record<string, string> = {
    not_started: 'Not evaluated',
    planned: 'Planned',
    running: 'Running',
    succeeded: 'Complete',
    failed: 'Failed',
    skipped: 'Skipped',
    unavailable: 'Unavailable',
  };
  return labels[normalized] || normalized;
}

function percent(value: unknown): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'Unknown';
  return `${Math.round(value * 100)}%`;
}

function bulletList(items: string[] | undefined, empty: string): string[] {
  if (!items || items.length === 0) return [`- ${empty}`];
  return items.map((item) => `- ${redactText(String(item))}`);
}

function findingLines(findings: Array<{ severity?: string; message?: string }> | undefined): string[] {
  if (!findings || findings.length === 0) return ['- No findings reported.'];
  return findings.map((finding) => {
    const severity = finding.severity ? `${finding.severity}: ` : '';
    return `- ${redactText(`${severity}${finding.message || ''}`)}`;
  });
}

export function buildEvaluationMarkdown(evaluation: Partial<PostTrainingEvaluation> | null | undefined): string {
  if (!evaluation) {
    return [
      '## Post-Training Evaluation',
      '',
      '**Status:** Not evaluated',
      '',
      'No static evaluation report is attached to this run yet.',
    ].join('\n');
  }
  const safe = redactJsonLike(evaluation) as Partial<PostTrainingEvaluation>;
  const scores = safe.scores || {};
  const artifacts = safe.artifact_paths?.length ? safe.artifact_paths : [safe.artifact_ref].filter(Boolean) as string[];
  const sections = [
    '## Post-Training Evaluation',
    '',
    `**Status:** ${evaluationStatusLabel(safe.status)}`,
    `**Provider:** \`${safe.provider || 'unknown'}\``,
    safe.domain ? `**Domain:** \`${safe.domain}\`` : '',
    safe.evaluation_type ? `**Mode:** \`${safe.evaluation_type}\`` : '',
    '',
    '| Score | Value |',
    '| --- | --- |',
    `| Overall | ${percent(scores.overall_score)} |`,
    `| Task relevance | ${percent(scores.task_relevance_score)} |`,
    `| Safety | ${percent(scores.safety_score)} |`,
    `| Privacy | ${percent(scores.privacy_score)} |`,
    `| Metric quality | ${percent(scores.metric_quality_score)} |`,
    `| Confidence | ${percent(scores.confidence)} |`,
    '',
    '### Recommendation',
    redactText(safe.recommendation || 'Human review is required before demo or client use.'),
    '',
    '### Quality Summary',
    redactText(safe.quality_summary || safe.failure_summary || 'No quality summary was reported.'),
    '',
    '### Test Prompts',
    ...bulletList(safe.test_prompts, 'No generated prompts were reported.'),
    '',
    '### Safety Findings',
    ...findingLines(safe.safety_findings),
    '',
    '### Privacy Findings',
    ...findingLines(safe.privacy_findings),
    '',
    '### Artifacts',
    ...bulletList(artifacts, 'No artifact link was reported.'),
    '',
    '### Report',
    redactText(safe.report_markdown || 'Static report unavailable.'),
  ].filter((line) => line !== '');
  return sections.join('\n');
}
