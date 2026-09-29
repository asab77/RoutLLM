import type { AdaptiveResponse } from '../api/types';

export const scenarios = [
  { id: 'normal', label: 'Threshold met', detail: 'A qualifying model is selected.' },
  { id: 'fallback', label: 'Routing fallback', detail: 'No model meets the routing threshold.' },
  { id: 'escalation', label: 'Escalation example', detail: 'Initial and returned models differ.' },
  { id: 'error', label: 'Service unavailable', detail: 'A controlled application error.' },
] as const;
export type Scenario = typeof scenarios[number]['id'];

const normal: AdaptiveResponse = {
  text: 'Adaptive routing chooses a model for each request, balancing predicted acceptability with projected cost.\n\nWhen several models meet the routing threshold, RoutLLM selects the least expensive qualifying model. If none qualify, the fallback policy chooses the candidate with the highest predicted acceptability.\n\nThe result: one interface for your request, with the routing decision available beneath each answer.',
  model_id: 'candidate-nemotron-3.5-lightning',
  provider: 'vercel',
  input_tokens: 12,
  output_tokens: 5,
  latency_ms: 184,
  estimated_cost_usd: '0.00000135',
  request_id: 'sim-threshold-001',
  routing: {
    selected_model_id: 'candidate-nemotron-3.5-lightning',
    threshold_satisfied: true,
    fallback_used: false,
    reason: 'quality_threshold_met',
  },
};

export const fixtures: Record<Exclude<Scenario, 'error'>, AdaptiveResponse> = {
  normal,
  fallback: {
    ...normal,
    text: 'A routing fallback is used when no candidate meets the required threshold. The policy selects the candidate with the highest predicted acceptability, using cost and model ID to break ties.\n\nFallback happens during initial model selection. It is separate from escalation, which can happen after a generated response fails validation.',
    model_id: 'candidate-claude-sonnet-5',
    input_tokens: 12,
    output_tokens: 5,
    latency_ms: 426,
    estimated_cost_usd: '0.000074',
    request_id: 'sim-fallback-001',
    routing: {
      selected_model_id: 'candidate-claude-sonnet-5',
      threshold_satisfied: false,
      fallback_used: true,
      reason: 'no_model_met_threshold_fallback',
    },
  },
  escalation: {
    ...normal,
    text: 'Validation checks whether a generated response meets an explicit output contract. If it fails, bounded escalation can try a different model.\n\nThe initial routing decision remains part of the record, even when another model returns the final answer. That distinction separates model selection from response validation.',
    model_id: 'candidate-gpt-6-luna',
    input_tokens: 12,
    output_tokens: 5,
    latency_ms: 240,
    estimated_cost_usd: '0.0000037',
    request_id: 'sim-escalation-001',
    execution: {
      attempts: 2,
      escalated: true,
      validation_outcome: 'passed',
      total_estimated_cost_usd: '0.00000505',
      total_latency_ms: 424,
    },
  },
};
