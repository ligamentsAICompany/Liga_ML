type DiscoveryRecord = Record<string, unknown>;

interface CandidateRecord {
  name: string;
  source: string;
  score: number | null;
  reason: string | null;
  url: string | null;
  license: string | null;
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

function isRecord(value: unknown): value is DiscoveryRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function valueLabel(value: unknown): string | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string') return value;
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

function normalizeCandidate(value: unknown): CandidateRecord | null {
  if (!isRecord(value)) return null;
  const score = getValue(value, ['score']);
  return {
    name: getString(value, ['name']) ?? 'Unnamed dataset',
    source: normalizeSource(getValue(value, ['source'])),
    score: typeof score === 'number' && Number.isFinite(score) ? Math.max(0, Math.min(score, 1)) : null,
    reason: getString(value, ['reason']),
    url: getString(value, ['url']),
    license: getString(value, ['license']),
    size: getString(value, ['size']),
    schemaHint: stringList(getValue(value, ['schemaHint', 'schema_hint'])),
    qualityNotes: stringList(getValue(value, ['qualityNotes', 'quality_notes'])),
    risks: stringList(getValue(value, ['risks'])),
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
  const parts = [`${candidate.name} (${sourceLabel(candidate.source)}, ${score})`];
  if (candidate.reason) parts.push(`Reason: ${candidate.reason}`);
  if (candidate.url) parts.push(`URL: ${candidate.url}`);
  if (candidate.license) parts.push(`License: ${candidate.license}`);
  if (candidate.size) parts.push(`Size: ${candidate.size}`);
  if (candidate.schemaHint.length) parts.push(`Schema: ${candidate.schemaHint.join(', ')}`);
  if (candidate.qualityNotes.length) parts.push(`Quality: ${candidate.qualityNotes.join(', ')}`);
  if (candidate.risks.length) parts.push(`Risks: ${candidate.risks.join(', ')}`);
  return parts.join(' · ');
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
      .sort((left, right) => (right.score ?? -1) - (left.score ?? -1))
    : [];

  const allowedSourceLines = allowedSources.map(sourceLabel);
  const excludedSourceLines = excludedSources.map((source) => (
    source === 'kaggle'
      ? 'Kaggle (future work only; not connected)'
      : sourceLabel(source)
  ));
  const candidateLines = candidates.length
    ? candidates.map(candidateSummary)
    : ['No candidate datasets supplied yet. Search allowed public sources, then inspect schema, license, privacy, and quality before training.'];
  const riskLines = candidates.flatMap((candidate) => candidate.risks);
  const summaryLines = [
    'No uploaded dataset is attached. Dataset discovery is required before training.',
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
