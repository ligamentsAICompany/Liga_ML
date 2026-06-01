import type { SessionMeta } from '../types/agent.js';

export type LLMErrorType =
  | 'quota'
  | 'billing'
  | 'auth'
  | 'rate_limit'
  | 'network'
  | 'empty_response'
  | 'unknown';

export interface LLMErrorInput {
  error: string;
  errorType?: string | null;
  model?: string | null;
  provider?: string | null;
  sessionId?: string | null;
  requestId?: string | null;
  turnId?: string | number | null;
  timestamp?: string;
}

export interface LLMErrorRecord {
  error: string;
  message: string;
  errorType: LLMErrorType;
  model: string;
  provider?: string | null;
  sessionId?: string | null;
  requestId?: string | null;
  turnId?: string | number | null;
  timestamp: string;
  transient: boolean;
  active: boolean;
}

export interface UnavailableModelRecord {
  model: string;
  errorType: LLMErrorType;
  message: string;
  timestamp: string;
}

export const LLM_ERROR_SELECT_MODEL_ACTION = 'Select another model';

export function normalizeLlmErrorType(raw?: string | null): LLMErrorType {
  const value = (raw ?? '').toLowerCase();
  if (value === 'quota' || value === 'quota_or_billing' || value === 'credits') return 'quota';
  if (value === 'billing') return 'billing';
  if (value === 'auth') return 'auth';
  if (value === 'rate_limit') return 'rate_limit';
  if (value === 'network') return 'network';
  if (value === 'empty_response') return 'empty_response';
  return 'unknown';
}

export function isTransientLlmError(errorType: LLMErrorType): boolean {
  return errorType === 'rate_limit' || errorType === 'network' || errorType === 'empty_response' || errorType === 'unknown';
}

export function modelSpecificErrorCopy(input: LLMErrorInput): string {
  const model = input.model || 'the selected model';
  const errorType = normalizeLlmErrorType(input.errorType);
  if (errorType === 'quota' || errorType === 'billing') {
    return `The model ${model} failed because its provider quota/spending limit was exceeded. Switch to another model or retry later.`;
  }
  if (errorType === 'auth') {
    return `The model ${model} failed because its provider credentials or permissions are invalid. Select another configured model or update credentials.`;
  }
  if (errorType === 'rate_limit') {
    return `The model ${model} was rate-limited by its provider. Retry later or select another model.`;
  }
  if (errorType === 'network') {
    return `The model ${model} could not be reached reliably. Retry later or select another model.`;
  }
  if (errorType === 'empty_response') {
    return `The model ${model} returned an empty response. Retry or select another model if it repeats.`;
  }
  return `The model ${model} failed before returning a usable response. Retry or select another model.`;
}

export function buildLlmErrorRecord(input: LLMErrorInput): LLMErrorRecord {
  const errorType = normalizeLlmErrorType(input.errorType);
  const model = input.model || 'unknown model';
  return {
    error: input.error || 'Unknown LLM error',
    message: modelSpecificErrorCopy({ ...input, model, errorType }),
    errorType,
    model,
    provider: input.provider ?? null,
    sessionId: input.sessionId ?? null,
    requestId: input.requestId ?? null,
    turnId: input.turnId ?? null,
    timestamp: input.timestamp ?? new Date().toISOString(),
    transient: isTransientLlmError(errorType),
    active: true,
  };
}

export function clearLlmErrorOnModelChange(
  current: LLMErrorRecord | null,
  nextModel: string,
): LLMErrorRecord | null {
  if (!current?.active) return current;
  if (current.model !== nextModel || current.transient) return null;
  return current;
}

export function clearLlmErrorOnNewPrompt(
  current: LLMErrorRecord | null,
  sessionId: string,
  nextRequestId: string,
): LLMErrorRecord | null {
  if (!current?.active) return current;
  if (current.sessionId === sessionId && current.requestId !== nextRequestId) return null;
  return current;
}

export function clearLlmErrorOnSuccessfulRequest(
  current: LLMErrorRecord | null,
  sessionId: string,
  requestId?: string | null,
): LLMErrorRecord | null {
  if (!current?.active) return current;
  if (current.sessionId !== sessionId) return current;
  if (!requestId || !current.requestId || current.requestId !== requestId) return null;
  return null;
}

export function shouldMarkModelUnavailable(errorType: LLMErrorType): boolean {
  return errorType === 'quota' || errorType === 'billing';
}

export function shouldReportLlmHealthErrorForActiveModel(
  input: Pick<LLMErrorInput, 'model'>,
  activeModel?: string | null,
): boolean {
  if (!activeModel || !input.model) return true;
  return input.model === activeModel;
}

export function markUnavailableModelForError<T extends SessionMeta>(
  session: T,
  input: LLMErrorInput | LLMErrorRecord,
): T {
  const errorType = normalizeLlmErrorType(input.errorType);
  const model = input.model || session.model;
  if (!model || !shouldMarkModelUnavailable(errorType)) return session;
  return {
    ...session,
    unavailableModels: {
      ...(session.unavailableModels ?? {}),
      [model]: {
        model,
        errorType,
        message: modelSpecificErrorCopy({ ...input, model, errorType }),
        timestamp: input.timestamp ?? new Date().toISOString(),
      },
    },
  };
}

type SessionModelAvailability = { unavailableModels?: SessionMeta['unavailableModels'] };

export function isModelUnavailable(session: SessionModelAvailability, model: string): boolean {
  return Boolean(session.unavailableModels?.[model]);
}

export function updateSessionModelPreservingContext<T extends SessionMeta>(
  session: T,
  model: string | null,
): T {
  return {
    ...session,
    model,
  };
}

export function suggestedFallbackModel(
  models: Array<{ modelPath: string }>,
  currentModel: string,
  session?: SessionModelAvailability,
): string | null {
  const fallback = models.find((model) => (
    model.modelPath !== currentModel && !isModelUnavailable(session ?? {}, model.modelPath)
  ));
  return fallback?.modelPath ?? null;
}
