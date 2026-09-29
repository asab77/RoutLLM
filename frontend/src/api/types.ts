export const categories = [
  'classification', 'coding', 'extraction', 'structured_json', 'qa',
  'reasoning', 'summarization',
] as const;
export type Category = typeof categories[number];

type JsonType = 'object' | 'array' | 'string' | 'number' | 'integer' | 'boolean' | 'null';
export interface ValidationContract {
  format: 'text' | 'json' | 'label';
  min_characters?: number | null;
  root_type?: JsonType | null;
  required_fields?: string[];
  field_types?: Record<string, JsonType>;
  allowed_labels?: string[];
}

export interface AdaptiveRequest {
  prompt: string;
  category: Category;
  system_prompt?: string | null;
  max_output_tokens?: number;
  temperature?: number;
  quality_threshold?: number | null;
  validation?: ValidationContract | null;
}

export interface AdaptiveResponse {
  text: string;
  model_id: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  estimated_cost_usd: string;
  request_id: string;
  routing: {
    selected_model_id: string;
    threshold_satisfied: boolean;
    fallback_used: boolean;
    reason: 'quality_threshold_met' | 'no_model_met_threshold_fallback';
  };
  execution?: {
    attempts: number;
    escalated: boolean;
    validation_outcome: 'passed';
    total_estimated_cost_usd?: string;
    total_latency_ms: number;
  };
}

export interface ApiErrorBody {
  error: { code: string; message: string };
  request_id?: string;
}

// A later relative-HTTP adapter can implement this boundary. No HTTP adapter exists now.
export interface InferenceClient {
  infer(request: AdaptiveRequest, options?: { signal?: AbortSignal }): Promise<AdaptiveResponse>;
}
