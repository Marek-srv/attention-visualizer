export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };

export type NumericVector = number[];
export type NumericMatrix = NumericVector[];
export type BooleanMatrix = boolean[][];
export type NullableNumericMatrix = Array<Array<number | null>>;

export interface HealthResponse {
  status: string;
  service: string;
}

export interface ToyWeights {
  query: NumericMatrix;
  key: NumericMatrix;
  value: NumericMatrix;
  /** The inspect route always resolves the default WO before responding. */
  output: NumericMatrix;
}

export interface ToyWeightsInput {
  query: NumericMatrix;
  key: NumericMatrix;
  value: NumericMatrix;
  output?: NumericMatrix | null;
}

export interface NormalizationParameters {
  gamma: NumericVector;
  beta: NumericVector;
  epsilon: number;
}

export interface NormalizationParametersInput {
  gamma?: NumericVector;
  beta?: NumericVector;
  epsilon?: number;
}

export interface FeedForwardParameters {
  input_weights: NumericMatrix;
  input_bias: NumericVector;
  output_weights: NumericMatrix;
  output_bias: NumericVector;
  normalization: NormalizationParameters;
}

export interface FeedForwardParametersInput {
  input_weights?: NumericMatrix;
  input_bias?: NumericVector;
  output_weights?: NumericMatrix;
  output_bias?: NumericVector;
  normalization?: NormalizationParametersInput;
}

export interface ToyInspectRequest {
  text: string;
  weights?: ToyWeightsInput;
  normalization?: NormalizationParametersInput;
  feed_forward?: FeedForwardParametersInput;
}

export interface ToyToken {
  token: string;
  normalized: string;
  token_id: number;
  position: number;
  token_embedding: NumericVector;
  position_embedding: NumericVector;
  combined_embedding: NumericVector;
}

export interface CalculationTerm {
  input_dimension: number;
  input_value: number;
  weight_value: number;
  product: number;
}

export interface DimensionBreakdown {
  output_dimension: number;
  terms: CalculationTerm[];
  result: number;
}

export interface ProjectionBreakdown {
  query: DimensionBreakdown[];
  key: DimensionBreakdown[];
  value: DimensionBreakdown[];
}

export interface ToyProjection {
  token: string;
  position: number;
  input_vector: NumericVector;
  query: NumericVector;
  key: NumericVector;
  value: NumericVector;
  breakdown: ProjectionBreakdown;
}

export interface AttentionProduct {
  dimension: number;
  query_value: number;
  key_value: number;
  product: number;
}

export interface AttentionCalculation {
  query_token: string;
  query_position: number;
  key_token: string;
  key_position: number;
  causally_masked: boolean;
  query_vector: NumericVector;
  key_vector: NumericVector;
  products: AttentionProduct[];
  raw_score: number;
  scaled_score: number;
  attention_weight: number;
}

export interface ContextTerm {
  key_token: string;
  key_position: number;
  attention_weight: number;
  value: number;
  product: number;
}

export interface ContextDimensionCalculation {
  output_dimension: number;
  terms: ContextTerm[];
  result: number;
}

export interface ContextCalculation {
  query_token: string;
  query_position: number;
  dimensions: ContextDimensionCalculation[];
}

export interface ToyAttention {
  key_dimension: number;
  scale_factor: number;
  /** True means the key is visible; false means it is causally masked. */
  causal_mask: BooleanMatrix;
  raw_scores: NumericMatrix;
  scaled_scores: NumericMatrix;
  masked_scores: NullableNumericMatrix;
  attention_weights: NumericMatrix;
  context_vectors: NumericMatrix;
  row_sums: NumericVector;
  calculations: AttentionCalculation[];
  context_calculations: ContextCalculation[];
}

export interface HeadAttention {
  head_index: number;
  dimension_indices: number[];
  query_vectors: NumericMatrix;
  key_vectors: NumericMatrix;
  value_vectors: NumericMatrix;
  raw_scores: NumericMatrix;
  scaled_scores: NumericMatrix;
  /** True means the key is visible; false means it is causally masked. */
  causal_mask: BooleanMatrix;
  masked_scores: NullableNumericMatrix;
  attention_weights: NumericMatrix;
  context_vectors: NumericMatrix;
  row_sums: NumericVector;
  calculations: AttentionCalculation[];
}

export interface OutputProjectionTerm {
  input_dimension: number;
  concatenated_input_value: number;
  weight_value: number;
  product: number;
}

export interface OutputDimensionCalculation {
  output_dimension: number;
  terms: OutputProjectionTerm[];
  result: number;
}

export interface OutputCalculation {
  token: string;
  position: number;
  dimensions: OutputDimensionCalculation[];
}

export interface MultiHeadAttention {
  model_dimension: number;
  number_of_heads: number;
  head_dimension: number;
  scale_factor: number;
  heads: HeadAttention[];
  concatenated_contexts: NumericMatrix;
  output_weight_matrix: NumericMatrix;
  projected_outputs: NumericMatrix;
  output_calculations: OutputCalculation[];
}

export interface ResidualTerm {
  dimension: number;
  input_value: number;
  attention_value: number;
  residual_value: number;
}

export interface NormalizationTerm {
  dimension: number;
  residual_value: number;
  mean: number;
  centered_value: number;
  standard_deviation: number;
  normalized_value: number;
  gamma: number;
  beta: number;
  output_value: number;
}

export interface NormalizationCalculation {
  token: string;
  position: number;
  input_vector: NumericVector;
  attention_output: NumericVector;
  residual_terms: ResidualTerm[];
  residual_vector: NumericVector;
  mean: number;
  variance: number;
  epsilon: number;
  standard_deviation: number;
  normalization_terms: NormalizationTerm[];
  normalized_vector: NumericVector;
  layer_norm_output: NumericVector;
}

export interface AttentionSublayer {
  architecture: string;
  input_vectors: NumericMatrix;
  attention_outputs: NumericMatrix;
  residual_vectors: NumericMatrix;
  gamma: NumericVector;
  beta: NumericVector;
  epsilon: number;
  normalized_vectors: NumericMatrix;
  layer_norm_outputs: NumericMatrix;
  calculations: NormalizationCalculation[];
}

export interface HiddenNeuronCalculation {
  hidden_neuron: number;
  terms: CalculationTerm[];
  weighted_sum: number;
  bias: number;
  pre_activation: number;
  activated: number;
  is_active: boolean;
}

export interface FeedForwardOutputCalculation {
  output_dimension: number;
  terms: CalculationTerm[];
  weighted_sum: number;
  bias: number;
  output: number;
}

export interface FeedForwardResidualTerm {
  dimension: number;
  input_value: number;
  feed_forward_value: number;
  residual_value: number;
}

export interface FeedForwardNormalizationTerm {
  dimension: number;
  normalized_value: number;
  gamma: number;
  beta: number;
  output_value: number;
}

export interface FeedForwardCalculation {
  token: string;
  position: number;
  input_vector: NumericVector;
  hidden_calculations: HiddenNeuronCalculation[];
  pre_activation_vector: NumericVector;
  activated_vector: NumericVector;
  output_calculations: FeedForwardOutputCalculation[];
  feed_forward_output: NumericVector;
  residual_terms: FeedForwardResidualTerm[];
  residual_vector: NumericVector;
  mean: number;
  variance: number;
  standard_deviation: number;
  normalized_vector: NumericVector;
  normalization_terms: FeedForwardNormalizationTerm[];
  transformer_block_output: NumericVector;
}

export interface FeedForwardSublayer {
  input_dimension: number;
  hidden_dimension: number;
  activation: string;
  input_vectors: NumericMatrix;
  input_weight_matrix: NumericMatrix;
  input_bias: NumericVector;
  pre_activation_vectors: NumericMatrix;
  activated_vectors: NumericMatrix;
  output_weight_matrix: NumericMatrix;
  output_bias: NumericVector;
  feed_forward_outputs: NumericMatrix;
  residual_vectors: NumericMatrix;
  normalization: NormalizationParameters;
  normalized_vectors: NumericMatrix;
  transformer_block_outputs: NumericMatrix;
  calculations: FeedForwardCalculation[];
}

export interface ToyInspectResponse {
  text: string;
  character_count: number;
  phase: number;
  token_count: number;
  vocabulary_size: number;
  embedding_dimension: number;
  tokens: ToyToken[];
  weights: ToyWeights;
  projections: ToyProjection[];
  attention: ToyAttention;
  multi_head_attention: MultiHeadAttention;
  attention_sublayer: AttentionSublayer;
  feed_forward_sublayer: FeedForwardSublayer;
}

export interface ModelConfigurationRequest {
  context_length?: number;
  d_model?: number;
  number_of_heads?: number;
  number_of_layers?: number;
  feed_forward_dimension?: number;
  dropout?: number;
}

export interface ModelConfiguration {
  vocab_size: number;
  context_length: number;
  d_model: number;
  number_of_heads: number;
  number_of_layers: number;
  feed_forward_dimension: number;
  dropout: number;
}

export interface TrainingConfiguration {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  weight_decay: number;
  gradient_clip: number;
  seed: number;
  validation_fraction: number;
  num_workers: number;
  device: "auto" | "cpu" | "cuda";
}

export interface TrainingStartRequest {
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  weight_decay?: number;
  gradient_clip?: number;
  seed?: number;
  validation_fraction?: number;
  model_config?: ModelConfigurationRequest | null;
}

export type TrainingJobState = "idle" | "running" | "completed" | "failed" | "cancelled";

export interface TrainingMetric {
  epoch: number;
  training_loss: number;
  validation_loss: number;
  training_perplexity: number;
  validation_perplexity: number;
  learning_rate: number;
  duration_seconds: number;
}

export interface TrainingStatus {
  state: TrainingJobState;
  status: TrainingJobState;
  job_id: string | null;
  current_epoch: number;
  total_epochs: number;
  latest_completed_epoch: number;
  latest_metrics: TrainingMetric | null;
  history: TrainingMetric[];
  best_validation_loss: number | null;
  cancellation_requested: boolean;
  checkpoint_file: string;
  checkpoint_available: boolean;
  error: string | null;
  model_config: ModelConfiguration | null;
  training_config: TrainingConfiguration | null;
}

export interface ModelLoadRequest {
  checkpoint_name?: string;
}

export interface ModelStatus {
  available: boolean;
  checkpoint_available: boolean;
  checkpoint_exists: boolean;
  checkpoint_file: string;
  metadata_file: string;
  metadata: JsonObject;
  loaded: boolean;
  loaded_checkpoint: string | null;
  device: string;
  architecture: string;
  model_config: ModelConfiguration | null;
  vocabulary_size: number | null;
  loaded_metadata: JsonObject;
}

export interface PredictionRequest {
  text: string;
  top_k?: number;
  temperature?: number;
}

export interface PredictionItem {
  token: string;
  token_id: number;
  logit: number;
  probability: number;
}

export interface PredictionResponse {
  input_text: string;
  tokens: string[];
  token_ids: number[];
  truncated: boolean;
  top_k: number;
  temperature: number;
  predictions: PredictionItem[];
  probability_sum: number;
  probability_label: string;
}

export type GenerationStrategy = "greedy" | "sample";

export interface GenerationRequest extends PredictionRequest {
  max_new_tokens?: number;
  strategy?: GenerationStrategy;
  seed?: number;
}

export interface GenerationStep {
  step: number;
  chosen_token: string;
  chosen_token_id: number;
  chosen_probability: number;
  top_predictions: PredictionItem[];
  is_eos: boolean;
}

export interface GenerationResponse {
  input_text: string;
  input_tokens: string[];
  input_token_ids: number[];
  truncated: boolean;
  generated_text: string;
  generated_tokens: string[];
  generated_token_ids: number[];
  strategy: GenerationStrategy;
  seed: number;
  temperature: number;
  top_k: number;
  max_new_tokens: number;
  stop_reason: "eos" | "max_new_tokens";
  steps: GenerationStep[];
  probability_label: string;
}

export interface TinyInspectRequest {
  text: string;
  layer?: number;
  head?: number;
  query_token?: number | null;
  key_token?: number | null;
  hidden_dimension?: number;
  top_k?: number;
}

export interface TinyLayerTrace {
  normalized_attention_input: NumericMatrix;
  query: NumericMatrix;
  key: NumericMatrix;
  value: NumericMatrix;
  raw_attention_scores: NumericMatrix;
  scaled_attention_scores: NumericMatrix;
  /** True means the key is visible; false means it is causally masked. */
  causal_mask: BooleanMatrix;
  attention_probabilities: NumericMatrix;
  head_context_vectors: NumericMatrix;
  concatenated_attention_output: NumericMatrix;
  projected_attention_output: NumericMatrix;
  attention_residual_output: NumericMatrix;
  normalized_feed_forward_input: NumericMatrix;
  feed_forward_pre_activations: NumericMatrix;
  gelu_activations: NumericMatrix;
  feed_forward_output: NumericMatrix;
  block_output: NumericMatrix;
}

export interface TinyInspectSelection {
  layer: number;
  head: number;
  query_token: number;
  key_token: number;
  hidden_dimension: number;
}

export interface SelectedAttentionCalculation {
  query_token: string;
  query_position: number;
  key_token: string;
  key_position: number;
  query_vector: NumericVector;
  key_vector: NumericVector;
  products: NumericVector;
  raw_score: number;
  scale_factor: number;
  scaled_score: number;
  causally_masked: boolean;
  attention_probability: number;
}

export interface TokenConnection {
  key_token: string;
  key_position: number;
  attention_weight: number;
  causally_available: boolean;
}

export interface SelectedHiddenValues {
  dimension: number;
  final_hidden_states: NumericVector;
}

export interface TinyInspectResponse {
  input_text: string;
  tokens: string[];
  token_ids: number[];
  truncated: boolean;
  architecture: string;
  attention_note: string;
  model_config: ModelConfiguration;
  selection: TinyInspectSelection;
  shapes: Record<string, number[]>;
  token_embeddings: NumericMatrix;
  position_embeddings: NumericMatrix;
  combined_embeddings: NumericMatrix;
  layer_trace: TinyLayerTrace;
  selected_attention_calculation: SelectedAttentionCalculation;
  token_connections: TokenConnection[];
  selected_hidden_values: SelectedHiddenValues;
  final_hidden_states: NumericMatrix;
  vocabulary_logits: NumericMatrix;
  vocabulary_probabilities: NumericMatrix;
  top_predictions: PredictionItem[];
  probability_sum: number;
}

export type PretrainedLoadState = "not_loaded" | "loading" | "loaded" | "failed";

export interface PretrainedModelDetails {
  name: string;
  device: string;
  number_of_layers: number;
  number_of_heads: number;
  hidden_dimension: number;
  vocabulary_size: number;
  context_length: number;
  attention_implementation: string;
}

export interface PretrainedStatus {
  status: PretrainedLoadState;
  loaded: boolean;
  loading: boolean;
  model_name: string;
  device: string;
  dependencies_available: {
    torch: boolean;
    transformers: boolean;
  };
  model: PretrainedModelDetails | null;
  error: string | null;
}

export interface PretrainedLoadRequest {
  model_name?: string | null;
}

export interface PretrainedInspectRequest {
  text: string;
  layer?: number;
  head?: number;
  query_token?: number | null;
  top_k?: number;
}

export interface PretrainedConnection {
  key_index: number;
  key_token: string;
  key_token_id: number;
  attention_weight: number;
  is_future: boolean;
}

export interface PretrainedInspectResponse {
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
  attention_shape: number[];
  attention_matrix: NumericMatrix;
  attention_row_sum: number;
  connections: PretrainedConnection[];
  top_predictions: PredictionItem[];
  probability_sum: number;
  attention_note: string;
}

export interface PretrainedPredictionResponse {
  model_name: string;
  device: string;
  input_text: string;
  tokens: string[];
  token_ids: number[];
  token_count: number;
  original_token_count: number;
  context_truncated: boolean;
  temperature: number;
  top_k: number;
  predictions: PredictionItem[];
  probability_sum: number;
  top_probability_sum: number;
  probability_label: string;
}

export interface ValidationIssue {
  type: string;
  loc: Array<string | number>;
  msg: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

export interface StructuredErrorDetail {
  code?: string;
  message: string;
}

export interface ApiErrorPayload {
  detail?: string | StructuredErrorDetail | ValidationIssue[];
  message?: string;
}
