export type UsageProviderId = 'hf-jobs' | 'gcp-vertex' | 'aws-sagemaker' | 'llm' | 'unknown';
export type CostConfidence = 'known' | 'estimated' | 'unknown';

export interface UsageEntry {
  usage_id: string;
  session_id: string;
  run_id?: string | null;
  provider: UsageProviderId | string;
  tool_name?: string | null;
  operation: string;
  job_id?: string | null;
  job_url?: string | null;
  artifact_url?: string | null;
  status: string;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  currency: string;
  estimated_cost_usd?: number | null;
  known_cost_usd?: number | null;
  cost_source?: string;
  cost_confidence?: CostConfidence | string;
  instance_type?: string | null;
  instance_count?: number | null;
  max_runtime_seconds?: number | null;
  actual_runtime_seconds?: number | null;
  dataset_name?: string | null;
  model_name?: string | null;
  output_policy?: string | null;
  approval_id?: string | null;
  approved: boolean;
  budget_cap_usd?: number | null;
  quota_status: string;
  warning?: string | null;
  error_summary?: string | null;
  metadata?: Record<string, unknown>;
}

export interface UsageSummary {
  total_estimated_cost_usd: number;
  total_known_cost_usd: number;
  cost_by_provider: Record<string, { estimated_cost_usd: number; known_cost_usd: number; count: number }>;
  cost_by_session: Record<string, { estimated_cost_usd: number; known_cost_usd: number; count: number }>;
  cost_by_run: Record<string, { estimated_cost_usd: number; known_cost_usd: number; count: number }>;
  recent_usage_entries: UsageEntry[];
  quota_warnings: Array<{ provider?: string; message?: string; usage_id?: string }>;
  budget_warnings: Array<{ provider?: string; message?: string; usage_id?: string }>;
  provider_readiness: Record<string, unknown>;
  usage_store?: {
    enabled: boolean;
    durable: boolean;
    store: string;
    warning?: string | null;
  } | null;
}
