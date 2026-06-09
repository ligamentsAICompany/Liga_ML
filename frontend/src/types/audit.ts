export type AuditCategory =
  | 'session'
  | 'dataset'
  | 'chat'
  | 'planner'
  | 'approval'
  | 'tool'
  | 'provider_job'
  | 'usage'
  | 'result'
  | 'error'
  | 'system'
  | 'security';

export type AuditSeverity = 'info' | 'warning' | 'error' | 'critical';

export interface AuditEvent {
  audit_id: string;
  session_id: string;
  run_id?: string | null;
  usage_id?: string | null;
  provider: string;
  event_type: string;
  category: AuditCategory | string;
  severity: AuditSeverity | string;
  status: string;
  title: string;
  message: string;
  timestamp?: string | null;
  actor: string;
  entity_type?: string | null;
  entity_id?: string | null;
  tool_name?: string | null;
  operation?: string | null;
  approval_id?: string | null;
  job_id?: string | null;
  job_url?: string | null;
  artifact_url?: string | null;
  dataset_name?: string | null;
  model_name?: string | null;
  output_policy?: string | null;
  estimated_cost_usd?: number | null;
  known_cost_usd?: number | null;
  error_code?: string | null;
  error_summary?: string | null;
  safe_metadata?: Record<string, unknown>;
}

export interface AuditStoreHealth {
  type: string;
  durable: boolean;
  enabled: boolean;
  warning?: string | null;
}

export interface AuditTimelineResponse {
  enabled: boolean;
  audit_store?: AuditStoreHealth | null;
  events: AuditEvent[];
}

export interface AuditSummary {
  enabled: boolean;
  total_events: number;
  counts_by_category: Record<string, number>;
  counts_by_severity: Record<string, number>;
  counts_by_provider: Record<string, number>;
  latest_warnings_errors: AuditEvent[];
  provider_job_timeline: AuditEvent[];
  approval_timeline: AuditEvent[];
  dataset_timeline: AuditEvent[];
  usage_cost_timeline: AuditEvent[];
  timeline_by_session: Record<string, AuditEvent[]>;
  timeline_by_run: Record<string, AuditEvent[]>;
  audit_store?: AuditStoreHealth | null;
}

export interface AuditFilters {
  provider?: string;
  category?: string;
  severity?: string;
  status?: string;
}
