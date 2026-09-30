import type { ChatResponse } from '../api/types';
import { RequestMetrics, formatCost } from './RequestMetrics';

const names: Record<string, string> = {
  'candidate-nemotron-3.5-lightning': 'Nemotron 3.5 Lightning',
  'candidate-claude-sonnet-5': 'Claude Sonnet 5',
  'candidate-gpt-6-luna': 'GPT-6 Luna',
};
export function modelName(id: string) { return names[id] ?? id; }

export function RoutingTrace({ response: r }: { response: ChatResponse }) {
  if (r.execution_mode === 'direct') {
    return <details className="routing-trace"><summary><span className="trace-title"><span className="trace-mark" aria-hidden="true">R↗</span><span><strong>Direct inference</strong><span>Returned by {modelName(r.model_id)}</span><small>{r.latency_ms} ms · {r.input_tokens} → {r.output_tokens} tokens · {formatCost(r.estimated_cost_usd)}</small></span></span><span className="route-action"><span className="when-closed">View route</span><span className="when-open">Hide route</span><span aria-hidden="true">⌄</span></span></summary>
      <div className="trace-expanded"><ol className="route-path" aria-label="Execution path"><li className="route-node endpoint">Request</li><li className="route-node router-node"><span className="small-label">DIRECT</span><strong>Direct inference</strong><span>Configured default model</span></li><li className="route-node selected-node"><span className="selected-label">RETURNED MODEL</span><strong>{modelName(r.model_id)}</strong><code>{r.model_id}</code></li><li className="route-node endpoint">Response</li></ol><RequestMetrics response={r} /></div>
    </details>;
  }

  const changed = r.routing.selected_model_id !== r.model_id;
  return <details className="routing-trace"><summary><span className="trace-title"><span className="trace-mark" aria-hidden="true">R↗</span><span><strong>Routed by RoutLLM</strong><span>Routed to {modelName(r.model_id)}</span><small>{r.latency_ms} ms · {r.input_tokens} → {r.output_tokens} tokens · {formatCost(r.estimated_cost_usd)}</small></span></span><span className="route-action"><span className="when-closed">View route</span><span className="when-open">Hide route</span><span aria-hidden="true">⌄</span></span></summary>
    <div className="trace-expanded"><ol className="route-path" aria-label="Routing path"><li className="route-node endpoint">Request</li><li className="route-node router-node"><span className="small-label">ROUTELLM ROUTER</span><strong>Adaptive routing</strong><span>Manual category: {r.category}</span><span className="policy-value">τ ≥ 0.80</span></li><li className="route-node selected-node"><span className="selected-label">{changed ? 'INITIAL MODEL' : 'SELECTED MODEL'}</span><strong>{modelName(r.routing.selected_model_id)}</strong><code>{r.routing.selected_model_id}</code><div className="route-flags"><span className={r.routing.threshold_satisfied ? 'success' : 'warning'}>{r.routing.threshold_satisfied ? '✓ Threshold met' : 'Threshold not met'}</span><span>{r.routing.fallback_used ? 'Routing fallback: Yes' : 'No routing fallback'}</span></div></li>{changed && <li className="route-node returned-node"><span className="selected-label">RETURNED MODEL</span><strong>{modelName(r.model_id)}</strong><code>{r.model_id}</code>{r.execution?.escalated && <span>Validation escalation</span>}</li>}<li className="route-node endpoint">Response</li></ol>
      <p className="route-reason">{r.routing.reason === 'quality_threshold_met' ? 'Quality threshold met' : 'No model met threshold; routing fallback selected'}<code>{r.routing.reason}</code></p>
      {r.execution && <p className="execution-note">{r.execution.attempts} attempts · Escalated: {r.execution.escalated ? 'Yes' : 'No'} · Validation: {r.execution.validation_outcome}</p>}
      <RequestMetrics response={r} />
    </div>
  </details>;
}
