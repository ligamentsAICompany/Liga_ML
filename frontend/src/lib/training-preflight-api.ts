import type { TrainingPreflightRequest, TrainingPreflightResult } from '../types/agent.js';
import { redactJsonLike, redactText } from './redaction.js';

export type RunTrainingPreflightRequest = Pick<TrainingPreflightRequest, 'session_id'> &
  Partial<Omit<TrainingPreflightRequest, 'session_id'>>;

async function readErrorText(response: Response): Promise<string> {
  const fallback = `${response.status} ${response.statusText}`.trim();
  try {
    const data = await response.json();
    if (data && typeof data === 'object' && 'detail' in data) {
      return redactText(String((data as { detail: unknown }).detail));
    }
    return redactText(JSON.stringify(data));
  } catch {
    try {
      return redactText(await response.text());
    } catch {
      return fallback;
    }
  }
}

async function expectPreflightResult(response: Response, options: { allowMissing?: boolean } = {}): Promise<TrainingPreflightResult | null> {
  const allowMissing = options.allowMissing === true;
  if (allowMissing && response.status === 404) return null;
  if (!response.ok) {
    const message = await readErrorText(response);
    throw new Error(`Training preflight API returned ${response.status}: ${message}`);
  }
  return (await response.json()) as TrainingPreflightResult;
}

function preflightFetch(path: string, options: RequestInit = {}): Promise<Response> {
  const headers = new Headers(options.headers);
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  return fetch(path, {
    ...options,
    headers,
    credentials: 'include',
  });
}

function safeBody(request: RunTrainingPreflightRequest): string {
  return JSON.stringify(redactJsonLike(request));
}

export async function runTrainingPreflight(request: RunTrainingPreflightRequest): Promise<TrainingPreflightResult> {
  const response = await preflightFetch('/api/training-preflight', {
    method: 'POST',
    body: safeBody(request),
  });
  return (await expectPreflightResult(response)) as TrainingPreflightResult;
}

export async function getLatestSessionPreflight(sessionId: string): Promise<TrainingPreflightResult | null> {
  const response = await preflightFetch(`/api/session/${encodeURIComponent(sessionId)}/preflight`);
  return expectPreflightResult(response, { allowMissing: true });
}

export async function getRunPreflight(sessionId: string, runId: string): Promise<TrainingPreflightResult | null> {
  const response = await preflightFetch(
    `/api/session/${encodeURIComponent(sessionId)}/runs/${encodeURIComponent(runId)}/preflight`,
  );
  return expectPreflightResult(response, { allowMissing: true });
}
