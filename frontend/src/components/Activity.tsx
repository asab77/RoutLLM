import { useCallback, useEffect, useRef, useState } from 'react';
import { normalizeError } from '../api/client';
import type { InferenceError } from '../api/client';
import type { ActivityItem, InferenceClient } from '../api/types';
import { formatCost } from './RequestMetrics';
import { modelName } from './RoutingTrace';

function modelPath(item: ActivityItem) {
  if (item.execution_mode === 'direct') {
    return modelName(item.final_model_id ?? item.initial_model_id ?? '—');
  }
  const initial = item.initial_model_id ? modelName(item.initial_model_id) : '—';
  const final = item.final_model_id ? modelName(item.final_model_id) : '—';
  return item.initial_model_id && item.final_model_id && item.initial_model_id !== item.final_model_id
    ? `${initial} → ${final}` : final !== '—' ? final : initial;
}

function cost(item: ActivityItem) {
  if (item.estimated_cost_usd === undefined) return '—';
  const formatted = formatCost(item.estimated_cost_usd);
  return item.cost_complete ? formatted : `${formatted} known`;
}

export function Activity({ client }: { client: InferenceClient }) {
  const [items, setItems] = useState<ActivityItem[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<InferenceError>();
  const busy = useRef(false);

  const load = useCallback(async (append: boolean, signal?: AbortSignal) => {
    if (busy.current) return;
    busy.current = true;
    setLoading(true);
    setError(undefined);
    try {
      const page = await client.activity({
        limit: 25,
        cursor: append && cursor ? cursor : undefined,
        signal,
      });
      setItems(previous => {
        const combined = append ? [...previous, ...page.items] : page.items;
        return combined.filter((item, index) =>
          combined.findIndex(candidate => candidate.request_id === item.request_id) === index
        );
      });
      setCursor(page.next_cursor ?? null);
    } catch (failure) {
      if (!signal?.aborted) setError(normalizeError(failure));
    } finally {
      busy.current = false;
      if (!signal?.aborted) setLoading(false);
    }
  }, [client, cursor]);

  useEffect(() => {
    const controller = new AbortController();
    void load(false, controller.signal);
    return () => controller.abort();
  }, [client]); // load intentionally excluded so cursor updates do not refresh.

  return <section className="activity activity-real" aria-label="Inference activity">
    <div className="activity-heading"><div><span className="small-label">REQUEST ACTIVITY</span><h1>Recent inference requests.</h1><p>Operational metadata only. Prompts and responses are not stored.</p></div><button className="text-button" disabled={loading} onClick={() => void load(false)}>Refresh</button></div>
    {loading && items.length === 0 ? <p role="status" className="activity-state">Loading activity…</p>
      : error && items.length === 0 ? <div className="activity-state error-panel" role="alert"><h2>Activity unavailable</h2><p>{error.message}</p><button className="text-button" onClick={() => void load(false)}>Try again</button></div>
      : items.length === 0 ? <div className="activity-state"><h2>No activity yet</h2><p>Completed chat executions will appear here.</p></div>
      : <>
        <div className="activity-table-wrap"><table className="activity-table"><thead><tr><th>Time</th><th>Mode</th><th>Category</th><th>Model</th><th>Status</th><th>Execution</th><th>Tokens</th><th>Latency</th><th>Cost</th></tr></thead><tbody>{items.map(item => <tr key={item.request_id}><td><time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString()}</time><code>{item.request_id}</code></td><td><span className="small-label">{item.execution_mode}</span></td><td>{item.execution_mode === 'adaptive' ? item.category ?? '—' : '—'}</td><td>{modelPath(item)}{item.provider && <small>{item.provider}</small>}</td><td><span className={item.outcome === 'returned' ? 'success' : 'warning'}>{item.outcome}</span></td><td>{item.attempt_count === undefined ? '—' : <>{item.attempt_count} attempt{item.attempt_count === 1 ? '' : 's'}{item.routing_fallback_used && <small>Routing fallback</small>}{item.escalated && <small>Validation escalation</small>}</>}</td><td>{item.input_tokens === undefined || item.output_tokens === undefined ? '—' : `${item.input_tokens} → ${item.output_tokens}`}</td><td>{item.latency_ms === undefined ? '—' : `${item.latency_ms} ms`}</td><td>{cost(item)}</td></tr>)}</tbody></table></div>
        {error && <p className="activity-inline-error" role="alert">{error.message}</p>}
        {cursor && <button className="text-button load-more" disabled={loading} onClick={() => void load(true)}>{loading ? 'Loading…' : 'Load more'}</button>}
      </>}
  </section>;
}
