export type JobState = "idle" | "running" | "completed" | "failed" | "cancelled";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type ModelConfig = {
  vocab_size?: number;
  context_length: number;
  d_model: number;
  number_of_heads: number;
  number_of_layers: number;
  feed_forward_dimension: number;
  dropout: number;
  vocabulary_size?: number;
};

export type TrainingMetric = {
  epoch: number;
  training_loss: number;
  validation_loss: number;
  training_perplexity: number;
  validation_perplexity: number;
  learning_rate: number;
  duration?: number;
  duration_seconds?: number;
};

export type TrainingStatus = {
  status: JobState;
  current_epoch?: number;
  latest_completed_epoch?: number;
  total_epochs?: number;
  history?: TrainingMetric[];
  metrics?: TrainingMetric[];
  best_validation_loss?: number | null;
  checkpoint_saved?: boolean;
  checkpoint_available?: boolean;
  error?: string | null;
  message?: string | null;
  model_config?: ModelConfig;
};

export type TrainingStartRequest = {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay?: number;
  gradient_clip?: number;
  seed?: number;
};

export type ModelStatus = {
  loaded: boolean;
  available?: boolean;
  checkpoint_available?: boolean;
  checkpoint_exists?: boolean;
  checkpoint_name?: string | null;
  checkpoint_file?: string | null;
  loaded_checkpoint?: string | null;
  device?: string;
  vocabulary_size?: number | null;
  current_epoch?: number | null;
  validation_loss?: number | null;
  model_config?: ModelConfig | null;
  error?: string | null;
  message?: string | null;
};

export type PredictionItem = {
  token: string;
  token_id: number;
  logit: number;
  temperature_adjusted_logit?: number;
  probability: number;
};

export type PredictionRequest = {
  text: string;
  top_k: number;
  temperature: number;
};

export type PredictionResponse = {
  input_text: string;
  tokens: string[];
  token_ids: number[];
  predictions: PredictionItem[];
  probability_sum: number;
  truncated?: boolean;
};

export type GenerationStrategy = "greedy" | "sample";

export type GenerationRequest = PredictionRequest & {
  max_new_tokens: number;
  strategy: GenerationStrategy;
  seed: number;
};

export type GenerationStep = {
  step: number;
  chosen_token: string;
  chosen_token_id: number;
  chosen_probability?: number;
  top_predictions: PredictionItem[];
  probability_sum?: number;
  is_eos?: boolean;
};

export type GenerationResponse = {
  input_text: string;
  generated_text: string;
  tokens?: string[];
  token_ids?: number[];
  steps: GenerationStep[];
  stop_reason?: string;
};

export type TinyInspectRequest = {
  text: string;
  layer: number;
  head: number;
  query_token: number;
  key_token?: number;
  hidden_dimension?: number;
  top_k: number;
};

export type TinyLayerTrace = {
  normalized_attention_input: number[][];
  query: number[][];
  key: number[][];
  value: number[][];
  raw_attention_scores: number[][];
  scaled_attention_scores: number[][];
  causal_mask: boolean[][];
  attention_probabilities: number[][];
  head_context_vectors: number[][];
  concatenated_attention_output: number[][];
  projected_attention_output: number[][];
  attention_residual_output: number[][];
  normalized_feed_forward_input: number[][];
  feed_forward_pre_activations: number[][];
  gelu_activations: number[][];
  feed_forward_output: number[][];
  block_output: number[][];
};

export type TinyInspectResponse = JsonObject & {
  input_text: string;
  tokens: string[];
  token_ids: number[];
  truncated: boolean;
  architecture: string;
  attention_note: string;
  model_config: ModelConfig;
  selection: { layer: number; head: number; query_token: number; key_token: number; hidden_dimension: number };
  shapes: { [key: string]: number[] };
  token_embeddings: number[][];
  position_embeddings: number[][];
  combined_embeddings: number[][];
  layer_trace: TinyLayerTrace;
  selected_attention_calculation: {
    query_token: string;
    query_position: number;
    key_token: string;
    key_position: number;
    query_vector: number[];
    key_vector: number[];
    products: number[];
    raw_score: number;
    scale_factor: number;
    scaled_score: number;
    causally_masked: boolean;
    attention_probability: number;
  };
  token_connections: Array<{ key_token: string; key_position: number; attention_weight: number; causally_available: boolean }>;
  selected_hidden_values: { dimension: number; final_hidden_states: number[] };
  final_hidden_states: number[][];
  vocabulary_logits: number[][];
  vocabulary_probabilities: number[][];
  top_predictions: PredictionItem[];
  probability_sum: number;
};

export type PretrainedStatus = {
  status: "not_loaded" | "loading" | "loaded" | "failed";
  loaded: boolean;
  loading?: boolean;
  model_name?: string;
  device?: string;
  dependencies_available?: { torch: boolean; transformers: boolean };
  model?: {
    name: string;
    device: string;
    number_of_layers: number;
    number_of_heads: number;
    hidden_dimension: number;
    vocabulary_size: number;
    context_length: number;
    attention_implementation: string;
  } | null;
  error?: string | null;
  message?: string | null;
};

export type PretrainedLoadRequest = {
  model_name?: string;
};

export type PretrainedInspectRequest = {
  text: string;
  layer: number;
  head: number;
  query_token?: number;
  top_k: number;
};

export type PretrainedConnection = {
  key_index: number;
  key_token: string;
  key_token_id: number;
  attention_weight: number;
  is_future: boolean;
};

export type PretrainedInspectResponse = {
  model_name: string;
  device: string;
  input_text: string;
  tokens: string[];
  token_ids: number[];
  token_count: number;
  original_token_count: number;
  context_truncated: boolean;
  selected_layer: number;
  selected_head: number;
  selected_query_index: number;
  selected_query_token: string;
  selected_query_token_id: number;
  attention_shape: [number, number];
  attention_matrix: number[][];
  attention_row_sum: number;
  connections: PretrainedConnection[];
  top_predictions: PredictionItem[];
  probability_sum: number;
  attention_note: string;
};
