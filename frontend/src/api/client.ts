import { fixtures } from '../mocks/fixtures';
import type { Scenario } from '../mocks/fixtures';
import type { ApiErrorBody, InferenceClient } from './types';

export type ErrorKind = 'authentication' | 'invalid_request' | 'rate_limited' |
  'unavailable' | 'deadline' | 'validation' | 'upstream' | 'network' | 'cancelled';

export class InferenceError extends Error {
  readonly kind: ErrorKind;
  readonly status?: number;
  readonly requestId?: string;
  readonly retryAfterSeconds?: number;

  constructor(
    kind: ErrorKind,
    message: string,
    status?: number,
    requestId?: string,
    retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = 'InferenceError';
    this.kind = kind;
    this.status = status;
    this.requestId = requestId;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

// Pure error translation, not a request sender. Never display raw edge/provider bodies.
export function fromHttpError(status: number, body?: ApiErrorBody, retryAfterSeconds?: number): InferenceError {
  const id = body?.request_id;
  switch (status) {
    case 401: return new InferenceError('authentication', 'Authentication is required to run inference.', status, id);
    case 422: return new InferenceError('invalid_request', 'Check the prompt, category, and output-token limit.', status, id);
    case 429: return new InferenceError('rate_limited', 'The inference rate limit was reached. Wait before submitting again.', status, id, retryAfterSeconds);
    case 503: return new InferenceError('unavailable', 'Inference is currently unavailable. No automatic retry will be made.', status, id);
    case 504: return new InferenceError('deadline', 'The inference deadline was exceeded. A timeout does not confirm that execution stopped.', status, id);
    case 502:
      return body?.error.code === 'response_validation_failed'
        ? new InferenceError('validation', 'No response passed the configured validation checks.', status, id)
        : new InferenceError('upstream', 'The upstream service could not complete the request.', status, id);
    default: return new InferenceError('upstream', 'The service returned an unexpected error.', status, id);
  }
}

export function normalizeError(error: unknown): InferenceError {
  return error instanceof InferenceError ? error : new InferenceError(
    'network', 'The request could not be completed. No automatic retry will be made.',
  );
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
          error: { code: 'adaptive_routing_unavailable', message: 'Simulated unavailable service' },
          request_id: 'sim-unavailable-001',
        });
      }
      // Clone so callers cannot mutate future runs; no prompts are retained by the adapter.
      return structuredClone(fixtures[scenario]);
    },
  };
}
