/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */
import type { AuditEvent } from './audit.js';

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
  cloudProvider?: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
}

export type CloudProviderId = 'hf-jobs' | 'gcp-vertex' | 'aws-sagemaker';
export type TrainingGoal = 'smoke-test' | 'production' | 'agent-decide';
export type OutputPolicy = 'cloud-private' | 'hf-hub' | 'cloud-and-hf-hub';

export type DatasetSourceFormat = 'csv' | 'json' | 'jsonl' | 'pdf' | 'docx' | 'xlsx' | 'md';

export interface UploadedDatasetInfo {
  repo_id?: string;
  repo_type: 'dataset';
  upload_id?: string;
  config_name?: string;
  filename?: string;
  raw_path_in_repo?: string;
  normalized_path_in_repo?: string;
  normalized_row_count?: number;
  normalized_format?: 'jsonl';
  source_format?: DatasetSourceFormat;
  source?: string;
  uploaded_at?: string | null;
  supports_training?: boolean;
  size_bytes?: number | null;
  format?: DatasetSourceFormat;
  status?: 'ready' | 'failed';
  hub_url?: string;
  load_dataset_snippet?: string;
}

export interface DatasetDiscoveryInfo {
  query?: string | null;
  intent?: Record<string, unknown>;
  allowed_sources?: string[];
  excluded_sources?: string[];
  candidates?: Array<Record<string, unknown>>;
  recommended_candidate?: Record<string, unknown> | null;
  warnings?: string[];
  selected_candidate?: Record<string, unknown> | null;
  timestamp?: string | null;
  requires_user_selection?: boolean;
}

export interface BackgroundRunProviderMetadata {
  provider: 'hf-jobs' | 'gcp-vertex' | 'aws-sagemaker' | 'none' | string;
  status?: string | null;
  job_id?: string | null;
  console_url?: string | null;
  logs_url?: string | null;
  artifact_path?: string | null;
  output_policy?: string | null;
  last_checked_at?: string | null;
  dataset_discovery?: DatasetDiscoveryInfo | null;
}

export interface BackgroundRunSummary {
  run_id: string;
  session_id: string;
  status: 'queued' | 'running' | 'waiting_approval' | 'waiting_provider' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted' | string;
  provider: 'hf-jobs' | 'gcp-vertex' | 'aws-sagemaker' | 'none' | string;
  created_at?: string | null;
  updated_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  last_event_seq?: number;
  active_tool?: string | null;
  active_provider_job_id?: string | null;
  approval_id?: string | null;
  error_summary?: string | null;
  result_summary?: string | null;
  provider_metadata?: BackgroundRunProviderMetadata;
  evaluation_id?: string | null;
  evaluation_status?: 'not_started' | 'planned' | 'running' | 'succeeded' | 'failed' | 'skipped' | 'unavailable' | string | null;
  evaluation_score?: number | null;
  audit_event_count?: number;
  audit_warning_count?: number;
  audit_error_count?: number;
  latest_audit_event?: AuditEvent | null;
  dataset_discovery?: DatasetDiscoveryInfo | null;
}

export interface UnavailableModelInfo {
  model: string;
  errorType: 'quota' | 'billing' | 'auth' | 'rate_limit' | 'network' | 'empty_response' | 'unknown';
  message: string;
  timestamp: string;
}

export interface DatasetUploadResponse extends UploadedDatasetInfo {
  session_id: string;
  private: true;
  path_in_repo: string;
  normalized_format: 'jsonl';
  size_bytes: number;
  source: string;
  uploaded_at: string;
}

export interface SessionMeta {
  id: string;
  title: string;
  createdAt: string;
  isActive: boolean;
  needsAttention: boolean;
  model?: string | null;
  cloudProvider?: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
  /** True when the backend no longer recognizes this session id (e.g.
   *  after a backend restart). The UI shows a recovery banner and
   *  disables input until the user chooses to restore-with-summary or
   *  start fresh. */
  expired?: boolean;
  autoApprovalEnabled?: boolean;
  autoApprovalCostCapUsd?: number | null;
  autoApprovalEstimatedSpendUsd?: number;
  autoApprovalRemainingUsd?: number | null;
  uploadedDatasets?: UploadedDatasetInfo[];
  latestDatasetDiscovery?: DatasetDiscoveryInfo | null;
  runs?: BackgroundRunSummary[];
  unavailableModels?: Record<string, UnavailableModelInfo>;
}

export interface ToolApproval {
  tool_call_id: string;
  approved: boolean;
  feedback?: string | null;
  namespace?: string | null;
}

export interface User {
  authenticated: boolean;
  username?: string;
  name?: string;
  picture?: string;
}
