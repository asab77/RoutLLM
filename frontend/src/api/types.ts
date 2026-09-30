export const categories = [
  'classification', 'coding', 'extraction', 'structured_json', 'qa',
  'reasoning', 'summarization',
] as const;
export type Category = typeof categories[number];

interface CommonChatRequest {
  prompt: string;
  system_prompt?: string | null;
  max_output_tokens?: number;
  temperature?: number;
}

export interface AutoChatRequest extends CommonChatRequest {
  routing_mode: 'auto';
  category?: never;
  validation?: never;
}

export interface ManualChatRequest extends CommonChatRequest {
  routing_mode: 'manual';
  category: Category;
  validation?: never;
}

export type ChatRequest = AutoChatRequest | ManualChatRequest;

export interface RoutingMetadata {
  selected_model_id: string;
  threshold_satisfied: boolean;
  fallback_used: boolean;
  reason: 'quality_threshold_met' | 'no_model_met_threshold_fallback';
}

export interface ExecutionMetadata {
  attempts: number;
  escalated: boolean;
  validation_outcome: 'passed';
  total_estimated_cost_usd?: string | null;
  total_latency_ms: number;
}

interface CommonChatResponse {
  text: string;
  model_id: string;
  provider: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  estimated_cost_usd: string;
  request_id: string;
}

export interface DirectChatResponse extends CommonChatResponse {
  execution_mode: 'direct';
  category?: never;
  category_source?: never;
  routing?: never;
  execution?: never;
}

export interface AdaptiveChatResponse extends CommonChatResponse {
  execution_mode: 'adaptive';
  category: Category;
  category_source: 'manual';
  routing: RoutingMetadata;
  execution?: ExecutionMetadata;
}

export type ChatResponse = DirectChatResponse | AdaptiveChatResponse;

export interface ActivityItem {
  request_id: string;
  created_at: string;
  execution_mode: 'direct' | 'adaptive';
  category?: Category;
  category_source?: 'manual';
  initial_model_id?: string;
  final_model_id?: string;
  provider?: string;
  routing_threshold_satisfied?: boolean;
  routing_fallback_used?: boolean;
  attempt_count?: number;
  escalated?: boolean;
  validation_outcome?: string;
  outcome: 'returned' | 'provider_failure' | 'validation_failed' | 'deadline_exceeded';
  error_category?: string;
  input_tokens?: number;
  output_tokens?: number;
  latency_ms?: number;
  estimated_cost_usd?: string;
  cost_complete: boolean;
}

export interface ActivityPage {
  items: ActivityItem[];
  next_cursor?: string | null;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
  request_id?: string;
}

export interface InferenceClient {
  infer(request: ChatRequest, options?: { signal?: AbortSignal }): Promise<ChatResponse>;
  activity(options?: { limit?: number; cursor?: string; signal?: AbortSignal }): Promise<ActivityPage>;
}
