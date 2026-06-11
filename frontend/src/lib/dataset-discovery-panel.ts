type DiscoveryRecord = Record<string, unknown>;

interface CandidateRecord {
  datasetId: string | null;
  title: string;
  source: string;
  score: number | null;
  relevanceScore: number | null;
  licenseScore: number | null;
  safetyScore: number | null;
  schemaScore: number | null;
  reason: string | null;
  url: string | null;
  repoId: string | null;
  license: string | null;
  licenseStatus: string | null;
  privacyStatus: string | null;
  schemaStatus: string | null;
  rowCount: number | null;
  columns: string[];
  textColumns: string[];
  labelColumns: string[];
  warnings: string[];
  excluded: boolean;
  exclusionReason: string | null;
  loadDatasetSnippet: string | null;
  size: string | null;
  schemaHint: string[];
  qualityNotes: string[];
  risks: string[];
}

export interface DatasetDiscoveryPanel {
  title: string;
  summaryLines: string[];
  allowedSourceLines: string[];
  excludedSourceLines: string[];
  candidateLines: string[];
  riskLines: string[];
  nextStepText: string;
  markdown: string;
}

const DEFAULT_ALLOWED_SOURCES = ['huggingface', 'github', 'papers', 'public_web'];
const DEFAULT_EXCLUDED_SOURCES = ['kaggle'];

const SOURCE_LABELS: Record<string, string> = {
  huggingface: 'Hugging Face Datasets',
  github: 'GitHub',
  papers: 'papers',
  public_web: 'public web',
  kaggle: 'Kaggle',
};

const REDACTED = '[REDACTED]';
const SECRET_RE = /\b(?:hf_[A-Za-z0-9_=-]+|sk-[A-Za-z0-9_-]+|[A-Za-z0-9_-]*secret[A-Za-z0-9_-]*)\b/gi;
const ENV_SECRET_RE = /\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)[A-Z0-9_]*)\s*=\s*[^ \n\t]+/gi;

function redactText(value: string): string {
  return value.replace(ENV_SECRET_RE, `$1=${REDACTED}`).replace(SECRET_RE, REDACTED);
}

function isRecord(value: unknown): value is DiscoveryRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function valueLabel(value: unknown): string | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string') return redactText(value);
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

function getValue(record: DiscoveryRecord, names: string[]): unknown {
  for (const name of names) {
    if (record[name] !== undefined) return record[name];
  }
  return undefined;
}

function getString(record: DiscoveryRecord, names: string[]): string | null {
  return valueLabel(getValue(record, names));
}

function normalizeSource(source: unknown): string {
  return (valueLabel(source) ?? 'huggingface').trim().toLowerCase().replace(/-/g, '_');
}

function sourceLabel(source: string): string {
  return SOURCE_LABELS[source] ?? source.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

function normalizeSourceList(value: unknown, fallback: string[]): string[] {
  const raw = Array.isArray(value) ? value : fallback;
  const normalized: string[] = [];
  for (const item of raw) {
    const source = normalizeSource(item);
    if (source && !normalized.includes(source)) normalized.push(source);
  }
  return normalized.length ? normalized : fallback;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(valueLabel).filter((item): item is string => !!item);
}

function riskList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (isRecord(item)) {
        return getString(item, ['message', 'reason', 'category']);
      }
      return valueLabel(item);
    })
    .filter((item): item is string => !!item);
}

function getNumber(record: DiscoveryRecord, names: string[]): number | null {
  const value = getValue(record, names);
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function scoreFrom(record: DiscoveryRecord, names: string[]): number | null {
  const direct = getNumber(record, names);
  if (direct !== null) return Math.max(0, Math.min(direct, 1));
  const quality = getValue(record, ['quality_score', 'qualityScore']);
  if (isRecord(quality)) {
    const nested = getNumber(quality, names);
    if (nested !== null) return Math.max(0, Math.min(nested, 1));
  }
  return null;
}

function normalizeCandidate(value: unknown): CandidateRecord | null {
  if (!isRecord(value)) return null;
  const score = scoreFrom(value, ['overall_score', 'overallScore', 'score']);
  const datasetId = getString(value, ['dataset_id', 'datasetId']);
  const title = getString(value, ['title', 'name']) ?? datasetId ?? 'Unnamed dataset';
  const rowCount = getNumber(value, ['row_count', 'rowCount']);
  return {
    datasetId,
    title,
    source: normalizeSource(getValue(value, ['source'])),
    score,
    relevanceScore: scoreFrom(value, ['relevance_score', 'relevanceScore']),
    licenseScore: scoreFrom(value, ['license_score', 'licenseScore']),
    safetyScore: scoreFrom(value, ['safety_score', 'safetyScore']),
    schemaScore: scoreFrom(value, ['schema_score', 'schemaScore']),
    reason: getString(value, ['reason']) ?? stringList(getValue(value, ['reasons']))[0] ?? null,
    url: getString(value, ['url', 'source_url', 'sourceUrl']),
    repoId: getString(value, ['repo_id', 'repoId']),
    license: getString(value, ['license']),
    licenseStatus: getString(value, ['license_status', 'licenseStatus']),
    privacyStatus: getString(value, ['privacy_status', 'privacyStatus']),
    schemaStatus: getString(value, ['schema_status', 'schemaStatus']),
    rowCount,
    columns: stringList(getValue(value, ['columns'])),
    textColumns: stringList(getValue(value, ['text_columns', 'textColumns'])),
    labelColumns: stringList(getValue(value, ['label_columns', 'labelColumns'])),
    warnings: stringList(getValue(value, ['warnings'])),
    excluded: getValue(value, ['excluded']) === true,
    exclusionReason: getString(value, ['exclusion_reason', 'exclusionReason']),
    loadDatasetSnippet: getString(value, ['load_dataset_snippet', 'loadDatasetSnippet']),
    size: getString(value, ['size']),
    schemaHint: stringList(getValue(value, ['schemaHint', 'schema_hint'])),
    qualityNotes: stringList(getValue(value, ['qualityNotes', 'quality_notes'])),
    risks: riskList(getValue(value, ['risks'])),
  };
}

function sectionLines(markdown: string, title: string): string[] {
  const pattern = new RegExp(`### ${title}\\s*\\n([\\s\\S]*?)(?=\\n### |\\n## |$)`, 'i');
  const match = markdown.match(pattern);
  if (!match) return [];
  return match[1]
    .split('\n')
    .map((line) => line.trim().replace(/^[-*]\s*/, ''))
    .filter(Boolean);
}

function parseMarkdownInput(markdown: string): DiscoveryRecord {
  const allowed = sectionLines(markdown, 'Allowed Sources');
  const excluded = sectionLines(markdown, 'Excluded Sources');
  const candidates = sectionLines(markdown, 'Candidate Ranking')
    .map((line) => {
      const match = line.match(/^\d+\.\s+\*\*(.*?)\*\*\s+\((.*?),\s+score\s+([0-9.]+)\)/i);
      if (!match) return null;
      return {
        name: match[1],
        source: match[2],
        score: Number(match[3]),
      };
    })
    .filter((item) => !!item);
  return {
    allowedSources: allowed,
    excludedSources: excluded,
    candidates,
    noUploadedDataset: /No uploaded dataset detected/i.test(markdown),
  };
}

function extractDiscoveryRecord(input: unknown): DiscoveryRecord {
  if (isRecord(input)) return input;
  if (typeof input === 'string') return parseMarkdownInput(input);
  return {};
}

function appendSection(lines: string[], title: string, items: string[]): void {
  if (!items.length) return;
  lines.push('', `### ${title}`, ...items.map((item) => `- ${item}`));
}

function candidateSummary(candidate: CandidateRecord): string {
  const score = candidate.score === null ? 'score not provided' : `score ${candidate.score.toFixed(2)}`;
  const badges = [
    candidate.excluded ? 'Excluded' : null,
    candidate.datasetId ? null : null,
  ].filter(Boolean);
  const parts = [`${candidate.title}${badges.length ? ` [${badges.join(', ')}]` : ''} (${sourceLabel(candidate.source)}, ${score})`];
  if (candidate.datasetId) parts.push(`Dataset ID: ${candidate.datasetId}`);
  if (candidate.reason) parts.push(`Reason: ${candidate.reason}`);
  if (candidate.url) parts.push(`URL: ${candidate.url}`);
  if (candidate.repoId) parts.push(`Repo: ${candidate.repoId}`);
  if (candidate.license) parts.push(`License: ${candidate.license}${candidate.licenseStatus ? ` (${candidate.licenseStatus})` : ''}`);
  if (candidate.licenseStatus) parts.push(`License ${candidate.licenseStatus}`);
  if (candidate.privacyStatus) parts.push(`Privacy: ${candidate.privacyStatus}`);
  if (candidate.privacyStatus) parts.push(`Privacy ${candidate.privacyStatus}`);
  if (candidate.schemaStatus) parts.push(`Schema: ${candidate.schemaStatus}`);
  if (candidate.schemaStatus) parts.push(`Schema ${candidate.schemaStatus}`);
  if (candidate.score !== null) {
    const scoreParts = [
      `Overall ${candidate.score.toFixed(2)}`,
      candidate.relevanceScore !== null ? `Relevance ${candidate.relevanceScore.toFixed(2)}` : null,
      candidate.licenseScore !== null ? `License ${candidate.licenseScore.toFixed(2)}` : null,
      candidate.safetyScore !== null ? `Privacy ${candidate.safetyScore.toFixed(2)}` : null,
      candidate.schemaScore !== null ? `Schema ${candidate.schemaScore.toFixed(2)}` : null,
    ].filter(Boolean);
    parts.push(`Scores: ${scoreParts.join(', ')}`);
  }
  if (candidate.rowCount !== null) parts.push(`Rows: ${candidate.rowCount.toLocaleString('en-US')}`);
  if (candidate.columns.length) parts.push(`Columns: ${candidate.columns.join(', ')}`);
  if (candidate.size) parts.push(`Size: ${candidate.size}`);
  if (candidate.schemaHint.length) parts.push(`Schema: ${candidate.schemaHint.join(', ')}`);
  if (candidate.qualityNotes.length) parts.push(`Quality: ${candidate.qualityNotes.join(', ')}`);
  if (candidate.warnings.length) parts.push(`Warnings: ${candidate.warnings.join(', ')}`);
  if (candidate.exclusionReason) parts.push(`Excluded: ${candidate.exclusionReason}`);
  if (candidate.loadDatasetSnippet) parts.push(`load_dataset: ${candidate.loadDatasetSnippet}`);
  if (candidate.risks.length) parts.push(`Risks: ${candidate.risks.join(', ')}`);
  return parts.join(' · ');
}

function intentLines(record: DiscoveryRecord): string[] {
  const intent = getValue(record, ['intent']);
  if (!isRecord(intent)) return [];
  const lines = [
    getString(intent, ['domain']) ? `Domain: ${getString(intent, ['domain'])}` : null,
    getString(intent, ['task_type', 'taskType']) ? `Task type: ${getString(intent, ['task_type', 'taskType'])}` : null,
    getString(intent, ['target_provider', 'targetProvider']) ? `Provider: ${getString(intent, ['target_provider', 'targetProvider'])}` : null,
    getString(intent, ['data_modality', 'dataModality']) ? `Modality: ${getString(intent, ['data_modality', 'dataModality'])}` : null,
  ].filter((line): line is string => !!line);
  return lines;
}

export function createDatasetDiscoveryPanel(input: unknown): DatasetDiscoveryPanel {
  const record = extractDiscoveryRecord(input);
  const allowedSources = normalizeSourceList(
    getValue(record, ['allowedSources', 'allowed_sources']),
    DEFAULT_ALLOWED_SOURCES,
  ).filter((source) => source !== 'kaggle');
  const excludedSources = normalizeSourceList(
    getValue(record, ['excludedSources', 'excluded_sources']),
    DEFAULT_EXCLUDED_SOURCES,
  );
  if (!excludedSources.includes('kaggle')) excludedSources.push('kaggle');

  const candidates = Array.isArray(getValue(record, ['candidates']))
    ? (getValue(record, ['candidates']) as unknown[])
      .map(normalizeCandidate)
      .filter((candidate): candidate is CandidateRecord => !!candidate)
      .sort((left, right) => {
        if (left.excluded !== right.excluded) return left.excluded ? 1 : -1;
        return (right.score ?? -1) - (left.score ?? -1);
      })
    : [];

  const allowedSourceLines = allowedSources.map(sourceLabel);
  const excludedSourceLines = excludedSources.map((source) => (
    source === 'kaggle'
      ? 'Kaggle (future work only; not connected)'
      : sourceLabel(source)
  ));
  const candidateLines = candidates.length
    ? candidates.map((candidate) => {
      const selected = isRecord(getValue(record, ['selected_candidate', 'selectedCandidate']))
        && getString(getValue(record, ['selected_candidate', 'selectedCandidate']) as DiscoveryRecord, ['dataset_id', 'datasetId']) === candidate.datasetId;
      const recommendedRecord = getValue(record, ['recommended_candidate', 'recommendedCandidate']);
      const recommended = isRecord(recommendedRecord)
        ? getString(recommendedRecord, ['dataset_id', 'datasetId']) === candidate.datasetId
        : !candidate.excluded && candidates.find((item) => !item.excluded) === candidate;
      const prefix = `${recommended ? 'Recommended · ' : ''}${selected ? 'Selected · ' : ''}`;
      return `${prefix}${candidateSummary(candidate)}`;
    })
    : [
      getString(record, ['no_candidates_reason', 'noCandidatesReason'])
        || 'No candidate datasets supplied yet. Search allowed public sources, then inspect schema, license, privacy, and quality before training.',
    ];
  const riskLines = candidates.flatMap((candidate) => [...candidate.risks, ...candidate.warnings]);
  const extractedIntentLines = intentLines(record);
  const summaryLines = [
    'No uploaded dataset is attached. Dataset discovery is required before training.',
    ...extractedIntentLines,
    `Allowed sources: ${allowedSourceLines.join(', ')}`,
    `Excluded sources: ${excludedSourceLines.join(', ')}`,
    'User selection required before training.',
  ];
  const nextStepText = 'User selection required before training. The planner does not download datasets, launch jobs, make cloud calls, or spend money.';

  const lines = [
    '## Dataset Discovery',
    '',
    ...summaryLines.map((line) => `- ${line}`),
  ];
  appendSection(lines, 'Allowed Sources', allowedSourceLines);
  appendSection(lines, 'Excluded Sources', excludedSourceLines);
  appendSection(lines, 'Extracted Intent', extractedIntentLines);
  appendSection(lines, 'Candidate Datasets', candidateLines);
  appendSection(lines, 'Risks', riskLines);
  lines.push('', nextStepText);

  return {
    title: 'Dataset Discovery',
    summaryLines,
    allowedSourceLines,
    excludedSourceLines,
    candidateLines,
    riskLines,
    nextStepText,
    markdown: lines.join('\n'),
  };
}
