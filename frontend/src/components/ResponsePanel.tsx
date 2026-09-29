import type { AdaptiveResponse, Category } from '../api/types';
import { RoutingTrace } from './RoutingTrace';
export function ResponsePanel({ response, category = 'qa', manual = false }: { response: AdaptiveResponse; category?: Category; manual?: boolean }) {
  return <article className="assistant-message" aria-label="RoutLLM response"><div className="assistant-heading"><span className="mark" aria-hidden="true">R↗</span><strong>RoutLLM</strong></div><div className="response-text">{response.text}</div><RoutingTrace response={response} category={category} manual={manual} /></article>;
}
