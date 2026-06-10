export const SESSION_CAPACITY_ACTION_MESSAGE =
  'You have too many active sessions. Clear stale inactive sessions or delete old sessions to continue.';

interface CapacityDetail {
  error?: unknown;
  message?: unknown;
  error_type?: unknown;
}

interface CapacityResponse {
  detail?: unknown;
}

export interface NormalizedSessionCapacityError {
  message: string;
  canCleanup: boolean;
}

function isCapacityDetail(value: unknown): value is CapacityDetail {
  return Boolean(value && typeof value === 'object');
}

export function normalizeSessionCapacityError(
  payload: CapacityResponse,
): NormalizedSessionCapacityError {
  const detail = payload.detail;
  if (isCapacityDetail(detail)) {
    const isCapacityError = detail.error === 'session_capacity' || detail.error_type === 'per_user';
    return {
      message: isCapacityError && typeof detail.message === 'string'
        ? SESSION_CAPACITY_ACTION_MESSAGE
        : SESSION_CAPACITY_ACTION_MESSAGE,
      canCleanup: isCapacityError,
    };
  }

  const rawMessage = typeof detail === 'string' ? detail : '';
  const looksLikeCapacity = rawMessage.toLowerCase().includes('capacity')
    || rawMessage.toLowerCase().includes('maximum');
  return {
    message: looksLikeCapacity ? SESSION_CAPACITY_ACTION_MESSAGE : SESSION_CAPACITY_ACTION_MESSAGE,
    canCleanup: looksLikeCapacity,
  };
}
