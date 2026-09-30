import { fixtures } from '../mocks/fixtures';
import type { Scenario } from '../mocks/fixtures';
import type { ActivityPage, ApiErrorBody, ChatResponse, InferenceClient } from './types';

export type ErrorKind = 'authentication' | 'invalid_request' | 'rate_limited' |
  'unavailable' | 'deadline' | 'validation' | 'upstream' | 'network' | 'cancelled';

export class InferenceError extends Error {
  readonly kind: ErrorKind;
  readonly status?: number;
  readonly requestId?: string;
  readonly retryAfterSeconds?: number;

  constructor(kind: ErrorKind, message: string, status?: number, requestId?: string, retryAfterSeconds?: number) {
    super(message);
    this.name = 'InferenceError';
    this.kind = kind;
    this.status = status;
    this.requestId = requestId;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

const allowedCodes = new Set([
  'invalid_request_id', 'invalid_request', 'invalid_adaptive_request',
  'rate_limit_exceeded', 'response_validation_failed', 'provider_failure',
  'provider_unavailable', 'adaptive_routing_unavailable', 'rate_limit_unavailable',
  'inference_deadline_exceeded', 'model_disabled', 'model_not_found',
]);

function safeBody(value: unknown): ApiErrorBody | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const body = value as Record<string, unknown>;
  const error = body.error;
  if (!error || typeof error !== 'object') return undefined;
  const detail = error as Record<string, unknown>;
  if (typeof detail.code !== 'string' || !allowedCodes.has(detail.code)) return undefined;
  return {
    error: { code: detail.code, message: '' },
    request_id: typeof body.request_id === 'string' ? body.request_id : undefined,
  };
}

export function fromHttpError(status: number, body?: ApiErrorBody, retryAfterSeconds?: number): InferenceError {
  const id = body?.request_id;
  switch (status) {
    case 400:
    case 422: return new InferenceError('invalid_request', 'Check the message and request settings.', status, id);
    case 401:
    case 403: return new InferenceError('authentication', 'Your session is not authorized to run this request.', status, id);
    case 404: return new InferenceError('invalid_request', 'The requested inference resource is unavailable.', status, id);
    case 429: return new InferenceError('rate_limited', 'Too many requests. Try again shortly.', status, id, retryAfterSeconds);
    case 503: return new InferenceError('unavailable', 'RoutLLM is temporarily unavailable.', status, id);
    case 504: return new InferenceError('deadline', 'The request took too long. Try again.', status, id);
    case 502:
      return body?.error.code === 'response_validation_failed'
        ? new InferenceError('validation', 'Output contract was not satisfied after the available attempts.', status, id)
        : new InferenceError('upstream', 'The model service could not complete the request.', status, id);
    default: return new InferenceError('upstream', 'RoutLLM returned an unexpected error.', status, id);
  }
}

export function normalizeError(error: unknown): InferenceError {
  if (error instanceof InferenceError) return error;
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new InferenceError('cancelled', 'The request was cancelled.');
  }
  return new InferenceError('network', 'RoutLLM could not be reached. Check your connection and try again.');
}

function retryAfter(response: Response): number | undefined {
  const raw = response.headers.get('Retry-After');
  if (raw === null || !/^\d+$/.test(raw.trim())) return undefined;
  const seconds = Number(raw);
  return Number.isSafeInteger(seconds) && seconds >= 0 ? seconds : undefined;
}

async function readErrorBody(response: Response): Promise<ApiErrorBody | undefined> {
  try {
    const text = await response.text();
    return text ? safeBody(JSON.parse(text)) : undefined;
  } catch {
    return undefined;
  }
}

function parseResponse(value: unknown): ChatResponse {
  if (!value || typeof value !== 'object') throw new InferenceError('upstream', 'RoutLLM returned an invalid response.');
  const body = value as Record<string, unknown>;
  if (body.execution_mode === 'direct' && typeof body.model_id === 'string') return body as unknown as ChatResponse;
  if (body.execution_mode === 'adaptive' && typeof body.model_id === 'string' && body.routing && typeof body.routing === 'object') return body as unknown as ChatResponse;
  throw new InferenceError('upstream', 'RoutLLM returned an invalid response.');
}

function parseActivityPage(value: unknown): ActivityPage {
  if (!value || typeof value !== 'object') throw new InferenceError('upstream', 'RoutLLM returned an invalid activity response.');
  const page = value as Record<string, unknown>;
  if (!Array.isArray(page.items)) throw new InferenceError('upstream', 'RoutLLM returned an invalid activity response.');
  return page as unknown as ActivityPage;
}

async function fetchJson(
  fetcher: typeof fetch,
  input: string,
  init?: RequestInit,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetcher(input, init);
  } catch (error) {
    throw normalizeError(error);
  }
  if (!response.ok) {
    throw fromHttpError(response.status, await readErrorBody(response), retryAfter(response));
  }
  try {
    return await response.json();
  } catch {
    throw new InferenceError('upstream', 'RoutLLM returned an invalid response.', response.status);
  }
}

export function createHttpClient(fetcher: typeof fetch = globalThis.fetch): InferenceClient {
  return {
    async infer(request, options) {
      const body = await fetchJson(fetcher, '/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(request),
          signal: options?.signal,
      });
      return parseResponse(body);
    },
    async activity(options) {
      const parameters = new URLSearchParams({ limit: String(options?.limit ?? 25) });
      if (options?.cursor) parameters.set('cursor', options.cursor);
      const body = await fetchJson(
        fetcher,
        `/v1/activity?${parameters.toString()}`,
        { signal: options?.signal },
      );
      return parseActivityPage(body);
    },
  };
}

function wait(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      clearTimeout(timer);
      reject(new InferenceError('cancelled', 'The simulation was cancelled.'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', cancel);
      resolve();
    }, 450);
    if (signal?.aborted) cancel();
    else signal?.addEventListener('abort', cancel, { once: true });
  });
}

export function createSimulatedClient(scenario: Scenario): InferenceClient {
  return {
    async infer(_request, options) {
      await wait(options?.signal);
      if (scenario === 'error') {
        throw fromHttpError(503, {
          error: { code: 'adaptive_routing_unavailable', message: '' },
          request_id: 'sim-unavailable-001',
        });
      }
      return structuredClone(fixtures[scenario]);
    },
    async activity() {
      return { items: [], next_cursor: null };
    },
  };
}
