/**
 * Agent-related types.
 *
 * Message and tool-call types are now provided by the Vercel AI SDK
 * (UIMessage, UIMessagePart, etc.). Only non-SDK types remain here.
 */

/** Custom metadata attached to every UIMessage via the `metadata` field. */
export interface MessageMeta {
  createdAt?: string;
  cloudProvider?: CloudProviderId;
  trainingGoal?: TrainingGoal;
  outputPolicy?: OutputPolicy;
}

export type CloudProviderId = 'hf-jobs' | 'gcp-vertex';
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
