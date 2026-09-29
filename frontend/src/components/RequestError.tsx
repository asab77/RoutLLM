import type { InferenceError } from '../api/client';

export function RequestError({ error }: { error: InferenceError }) {
  return <section className="error-panel" role="alert">
    <span className="small-label">REQUEST ERROR {error.status}</span>
    <h2>Request could not complete</h2><p>{error.message}</p>
    {error.retryAfterSeconds !== undefined && <p>Wait at least {error.retryAfterSeconds} seconds before a manual retry.</p>}
    {error.requestId && <p className="request-id">Request ID <code>{error.requestId}</code></p>}
  </section>;
}
