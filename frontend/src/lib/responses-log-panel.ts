export interface ResponsesSummary {
  total_responses: number;
  visible_count: number;
  batch_number: number;
  has_rows: boolean;
  button_enabled: boolean;
  durable?: boolean;
  store_type?: string;
}

export interface ResponseLogRow {
  id?: string;
  display_session_number: number;
  actual_sequence_number: number;
  batch_number: number;
  session_id: string;
  short_session_id?: string | null;
  session_title?: string | null;
  model_name: string;
  platform: string;
  run_type: string;
  result_storage: string;
  progress: string;
  job_id: string;
  final_artifact_or_result: string;
  created_at?: string | null;
  completed_at?: string | null;
  provider_metadata?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ResponsesPagePayload {
  rows: ResponseLogRow[];
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface ResponsesQueryState {
  page: number;
  pageSize: number;
  platform?: string;
  progress?: string;
  model?: string;
  sessionId?: string;
  jobId?: string;
  q?: string;
}

export interface ResponseColumn {
  key: string;
  label: string;
}

export const REQUIRED_RESPONSE_COLUMNS: ResponseColumn[] = [
  { key: 'session', label: 'Session Number' },
  { key: 'model', label: 'Model Name' },
  { key: 'platform', label: 'Platform' },
  { key: 'runType', label: 'Run Type' },
  { key: 'storage', label: 'Result Storage' },
  { key: 'progress', label: 'Progress' },
  { key: 'jobId', label: 'Job ID' },
  { key: 'result', label: 'Final Artifact / Result' },
];

export const OPTIONAL_RESPONSE_COLUMNS: ResponseColumn[] = [
  { key: 'createdAt', label: 'Created At' },
  { key: 'completedAt', label: 'Completed At' },
  { key: 'shortSessionId', label: 'Short Session ID' },
];

const SECRET_PATTERNS = [
  /(Bearer\s+)[A-Za-z0-9._\-+/=]+/gi,
  /hf_[A-Za-z0-9_]{8,}/g,
  /(token|secret|password|api[_-]?key|access[_-]?key)(\s*[=:]\s*)([^\s,;&]+)/gi,
];

export function redactResponseText(value: unknown): string {
  let text = value == null ? '' : String(value);
  for (const pattern of SECRET_PATTERNS) {
    text = text.replace(pattern, (...args: string[]) => {
      const match = args[0];
      if (/^bearer\s+/i.test(match)) {
        return `${args[1]}[REDACTED]`;
      }
      if (/^(token|secret|password|api[_-]?key|access[_-]?key)/i.test(match)) {
        return `${args[1]}${args[2]}[REDACTED]`;
      }
      return '[REDACTED]';
    });
  }
  return text;
}

function sessionLabel(row: ResponseLogRow): string {
  const base = String(row.display_session_number || '');
  if (row.actual_sequence_number && row.batch_number) {
    return `${base} (actual ${row.actual_sequence_number}, batch ${row.batch_number})`;
  }
  return base;
}

function clean(value: unknown, fallback = '-'): string {
  const text = redactResponseText(value).trim();
  return text || fallback;
}

export function createResponsesButtonState({
  summary,
}: {
  isProcessing: boolean;
  summary: ResponsesSummary | null;
}) {
  return {
    visible: true,
    disabled: summary?.button_enabled === false ? false : false,
    label: summary?.has_rows ? `Responses (${summary.visible_count})` : 'Responses',
  };
}

export function createResponsesQueryParams({
  page,
  pageSize,
  platform,
  progress,
  model,
  sessionId,
  jobId,
  q,
}: ResponsesQueryState): URLSearchParams {
  const params = new URLSearchParams();
  params.set('page', String(Math.max(1, page || 1)));
  params.set('page_size', String(Math.max(1, pageSize || 50)));
  for (const [key, value] of [
    ['platform', platform],
    ['progress', progress],
    ['model', model],
    ['session_id', sessionId],
    ['job_id', jobId],
    ['q', q],
  ] as const) {
    const cleanValue = String(value || '').trim();
    if (cleanValue) {
      params.set(key, cleanValue);
    }
  }
  return params;
}

export function createResponsesPaginationModel(payload: ResponsesPagePayload | null) {
  const page = payload?.page || 1;
  const rowCount = payload?.rows?.length || 0;
  const totalRows = payload?.total_rows || rowCount;
  const pageSize = payload?.page_size || Math.max(rowCount, 1);
  const totalPages =
    payload?.total_pages && payload.total_pages > 0
      ? payload.total_pages
      : totalRows > 0
        ? Math.max(1, Math.ceil(totalRows / pageSize))
        : 0;
  return {
    label: totalPages > 0 ? `Page ${page} of ${totalPages} • ${totalRows} responses` : '0 responses',
    canGoPrevious: Boolean(payload?.has_previous),
    canGoNext: Boolean(payload?.has_next),
    previousPage: Math.max(1, page - 1),
    nextPage: page + 1,
  };
}

export function createResponsesPanelModel({
  rows,
  error = null,
  filtersActive = false,
}: {
  rows: ResponseLogRow[];
  error?: string | null;
  filtersActive?: boolean;
}) {
  return {
    columns: [...REQUIRED_RESPONSE_COLUMNS, ...OPTIONAL_RESPONSE_COLUMNS],
    emptyStateTitle: filtersActive ? 'No responses match your filters' : 'No responses yet',
    emptyStateDescription: filtersActive
      ? 'Clear the filters or search text to see all persisted Responses rows.'
      : 'Fine-tuning and cloud job outcomes will appear here after a run starts or finishes.',
    errorMessage: error,
    rows: rows.map((row) => ({
      raw: row,
      cells: {
        session: sessionLabel(row),
        model: clean(row.model_name),
        platform: clean(row.platform),
        runType: clean(row.run_type),
        storage: clean(row.result_storage),
        progress: clean(row.progress),
        jobId: clean(row.job_id),
        result: clean(row.final_artifact_or_result),
        createdAt: clean(row.created_at),
        completedAt: clean(row.completed_at),
        shortSessionId: clean(row.short_session_id || row.session_id?.slice(0, 8)),
      },
    })),
  };
}
