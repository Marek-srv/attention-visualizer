import { FormEvent, useState } from "react";

type EmbeddingKind = "token_embedding" | "position_embedding" | "combined_embedding";
type ProjectionKind = "query" | "key" | "value";
type AttentionView = "raw_scores" | "scaled_scores" | "masked_scores" | "attention_weights";
type Matrix = number[][];
type Weights = Record<ProjectionKind, Matrix> & { output: Matrix };
type NormalizationParameters = { gamma: number[]; beta: number[]; epsilon: number };
type FeedForwardParameters = {
  input_weights: Matrix;
  input_bias: number[];
  output_weights: Matrix;
  output_bias: number[];
  normalization: NormalizationParameters;
};
type FeedForwardEditor = "input_weights" | "input_bias" | "output_weights" | "output_bias" | "normalization";

type TokenResult = {
  token: string;
  normalized: string;
  token_id: number;
  position: number;
  token_embedding: number[];
  position_embedding: number[];
  combined_embedding: number[];
};

type CalculationTerm = {
  input_dimension: number;
  input_value: number;
  weight_value: number;
  product: number;
};

type DimensionBreakdown = {
  output_dimension: number;
  terms: CalculationTerm[];
  result: number;
};

type ProjectionResult = {
  token: string;
  position: number;
  input_vector: number[];
  query: number[];
  key: number[];
  value: number[];
  breakdown: Record<ProjectionKind, DimensionBreakdown[]>;
};

type AttentionProduct = { dimension: number; query_value: number; key_value: number; product: number };
type AttentionCalculation = {
  query_token: string;
  query_position: number;
  key_token: string;
  key_position: number;
  causally_masked: boolean;
  query_vector: number[];
  key_vector: number[];
  products: AttentionProduct[];
  raw_score: number;
  scaled_score: number;
  attention_weight: number;
};
type ContextTerm = { key_token: string; key_position: number; attention_weight: number; value: number; product: number };
type ContextCalculation = {
  query_token: string;
  query_position: number;
  dimensions: Array<{ output_dimension: number; terms: ContextTerm[]; result: number }>;
};
type AttentionResult = {
  key_dimension: number;
  scale_factor: number;
  causal_mask: boolean[][];
  raw_scores: number[][];
  scaled_scores: number[][];
  masked_scores: Array<Array<number | null>>;
  attention_weights: number[][];
  context_vectors: number[][];
  row_sums: number[];
  calculations: AttentionCalculation[];
  context_calculations: ContextCalculation[];
};
type HeadAttention = {
  head_index: number;
  dimension_indices: number[];
  query_vectors: number[][];
  key_vectors: number[][];
  value_vectors: number[][];
  raw_scores: number[][];
  scaled_scores: number[][];
  causal_mask: boolean[][];
  masked_scores: Array<Array<number | null>>;
  attention_weights: number[][];
  context_vectors: number[][];
  row_sums: number[];
  calculations: AttentionCalculation[];
};
type OutputCalculation = {
  token: string;
  position: number;
  dimensions: Array<{
    output_dimension: number;
    terms: Array<{ input_dimension: number; concatenated_input_value: number; weight_value: number; product: number }>;
    result: number;
  }>;
};
type MultiHeadAttention = {
  model_dimension: number;
  number_of_heads: number;
  head_dimension: number;
  scale_factor: number;
  heads: HeadAttention[];
  concatenated_contexts: number[][];
  output_weight_matrix: Matrix;
  projected_outputs: number[][];
  output_calculations: OutputCalculation[];
};
type ResidualTerm = { dimension: number; input_value: number; attention_value: number; residual_value: number };
type NormalizationTerm = {
  dimension: number;
  residual_value: number;
  mean: number;
  centered_value: number;
  standard_deviation: number;
  normalized_value: number;
  gamma: number;
  beta: number;
  output_value: number;
};
type NormalizationCalculation = {
  token: string;
  position: number;
  input_vector: number[];
  attention_output: number[];
  residual_terms: ResidualTerm[];
  residual_vector: number[];
  mean: number;
  variance: number;
  epsilon: number;
  standard_deviation: number;
  normalization_terms: NormalizationTerm[];
  normalized_vector: number[];
  layer_norm_output: number[];
};
type AttentionSublayer = {
  architecture: "post_norm";
  input_vectors: number[][];
  attention_outputs: number[][];
  residual_vectors: number[][];
  gamma: number[];
  beta: number[];
  epsilon: number;
  normalized_vectors: number[][];
  layer_norm_outputs: number[][];
  calculations: NormalizationCalculation[];
};
type FeedForwardCalculation = {
  token: string;
  position: number;
  input_vector: number[];
  hidden_calculations: Array<{ hidden_neuron: number; terms: Array<{ input_dimension: number; input_value: number; weight_value: number; product: number }>; weighted_sum: number; bias: number; pre_activation: number; activated: number; is_active: boolean }>;
  pre_activation_vector: number[];
  activated_vector: number[];
  output_calculations: Array<{ output_dimension: number; terms: Array<{ input_dimension: number; input_value: number; weight_value: number; product: number }>; weighted_sum: number; bias: number; output: number }>;
  feed_forward_output: number[];
  residual_terms: Array<{ dimension: number; input_value: number; feed_forward_value: number; residual_value: number }>;
  residual_vector: number[];
  mean: number;
  variance: number;
  standard_deviation: number;
  normalized_vector: number[];
  normalization_terms: Array<{ dimension: number; normalized_value: number; gamma: number; beta: number; output_value: number }>;
  transformer_block_output: number[];
};
type FeedForwardSublayer = {
  input_dimension: number;
  hidden_dimension: number;
  activation: "relu";
  input_vectors: number[][];
  input_weight_matrix: Matrix;
  input_bias: number[];
  pre_activation_vectors: number[][];
  activated_vectors: number[][];
  output_weight_matrix: Matrix;
  output_bias: number[];
  feed_forward_outputs: number[][];
  residual_vectors: number[][];
  normalization: NormalizationParameters;
  normalized_vectors: number[][];
  transformer_block_outputs: number[][];
  calculations: FeedForwardCalculation[];
};

type InspectResult = {
  text: string;
  character_count: number;
  phase: number;
  token_count: number;
  vocabulary_size: number;
  embedding_dimension: number;
  tokens: TokenResult[];
  weights: Weights;
  projections: ProjectionResult[];
  attention: AttentionResult;
  multi_head_attention: MultiHeadAttention;
  attention_sublayer: AttentionSublayer;
  feed_forward_sublayer: FeedForwardSublayer;
};

const embeddingLabels: Record<EmbeddingKind, string> = {
  token_embedding: "Token",
  position_embedding: "Position",
  combined_embedding: "Combined",
};

const projectionLabels: Record<ProjectionKind, string> = {
  query: "WQ — Query weights",
  key: "WK — Key weights",
  value: "WV — Value weights",
};

const projectionSymbols: Record<ProjectionKind, string> = { query: "Q", key: "K", value: "V" };
const attentionViewLabels: Record<AttentionView, string> = {
  raw_scores: "Raw scores",
  scaled_scores: "Scaled scores",
  masked_scores: "Masked scores",
  attention_weights: "Attention weights",
};

function cloneWeights(weights: Weights): Weights {
  return {
    query: weights.query.map((row) => [...row]),
    key: weights.key.map((row) => [...row]),
    value: weights.value.map((row) => [...row]),
    output: weights.output.map((row) => [...row]),
  };
}

function cloneFeedForward(parameters: FeedForwardParameters): FeedForwardParameters {
  return {
    input_weights: parameters.input_weights.map((row) => [...row]),
    input_bias: [...parameters.input_bias],
    output_weights: parameters.output_weights.map((row) => [...row]),
    output_bias: [...parameters.output_bias],
    normalization: {
      gamma: [...parameters.normalization.gamma],
      beta: [...parameters.normalization.beta],
      epsilon: parameters.normalization.epsilon,
    },
  };
}

function heatColor(value: number): string {
  const strength = Math.min(Math.abs(value), 1);
  return value >= 0
    ? `rgba(116, 235, 178, ${0.1 + strength * 0.68})`
    : `rgba(244, 119, 119, ${0.1 + strength * 0.62})`;
}

function vectorMean(vector: number[]): number {
  return vector.reduce((sum, value) => sum + value, 0) / vector.length;
}

function vectorVariance(vector: number[]): number {
  const mean = vectorMean(vector);
  return vector.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vector.length;
}

export default function ToyMathLab() {
  const [text, setText] = useState("I love");
  const [result, setResult] = useState<InspectResult | null>(null);
  const [weights, setWeights] = useState<Weights | null>(null);
  const [defaultWeights, setDefaultWeights] = useState<Weights | null>(null);
  const [normalization, setNormalization] = useState<NormalizationParameters | null>(null);
  const [defaultNormalization, setDefaultNormalization] = useState<NormalizationParameters | null>(null);
  const [feedForward, setFeedForward] = useState<FeedForwardParameters | null>(null);
  const [defaultFeedForward, setDefaultFeedForward] = useState<FeedForwardParameters | null>(null);
  const [embeddingKind, setEmbeddingKind] = useState<EmbeddingKind>("combined_embedding");
  const [matrixKind, setMatrixKind] = useState<ProjectionKind>("query");
  const [projectionKind, setProjectionKind] = useState<ProjectionKind>("query");
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [outputDimension, setOutputDimension] = useState(0);
  const [attentionView, setAttentionView] = useState<AttentionView>("attention_weights");
  const [selectedKeyPosition, setSelectedKeyPosition] = useState(0);
  const [contextDimension, setContextDimension] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [multiHeadOutputDimension, setMultiHeadOutputDimension] = useState(0);
  const [normalizationDimension, setNormalizationDimension] = useState(0);
  const [selectedHiddenNeuron, setSelectedHiddenNeuron] = useState(0);
  const [feedForwardOutputDimension, setFeedForwardOutputDimension] = useState(0);
  const [feedForwardEditor, setFeedForwardEditor] = useState<FeedForwardEditor>("input_weights");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function inspect(customWeights?: Weights, customNormalization?: NormalizationParameters, customFeedForward?: FeedForwardParameters) {
    setError("");
    if (!text.trim()) {
      setError("Enter some text before starting the inspection.");
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, ...(customWeights ? { weights: customWeights } : {}), ...(customNormalization ? { normalization: customNormalization } : {}), ...(customFeedForward ? { feed_forward: customFeedForward } : {}) }),
      });
      if (!response.ok) {
        const detail = (await response.json()) as { detail?: Array<{ msg?: string }> };
        throw new Error(detail.detail?.[0]?.msg ?? "The inspection request failed.");
      }
      const data = (await response.json()) as InspectResult;
      setResult(data);
      setSelectedPosition(0);
      setWeights(cloneWeights(data.weights));
      if (!customWeights) setDefaultWeights(cloneWeights(data.weights));
      const returnedNormalization = {
        gamma: [...data.attention_sublayer.gamma],
        beta: [...data.attention_sublayer.beta],
        epsilon: data.attention_sublayer.epsilon,
      };
      setNormalization(returnedNormalization);
      if (!customNormalization) setDefaultNormalization(returnedNormalization);
      const returnedFeedForward = {
        input_weights: data.feed_forward_sublayer.input_weight_matrix.map((row) => [...row]),
        input_bias: [...data.feed_forward_sublayer.input_bias],
        output_weights: data.feed_forward_sublayer.output_weight_matrix.map((row) => [...row]),
        output_bias: [...data.feed_forward_sublayer.output_bias],
        normalization: {
          gamma: [...data.feed_forward_sublayer.normalization.gamma],
          beta: [...data.feed_forward_sublayer.normalization.beta],
          epsilon: data.feed_forward_sublayer.normalization.epsilon,
        },
      };
      setFeedForward(returnedFeedForward);
      if (!customFeedForward) setDefaultFeedForward(returnedFeedForward);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void inspect();
  }

  function updateWeight(row: number, column: number, value: number) {
    if (!weights || !Number.isFinite(value)) return;
    const next = cloneWeights(weights);
    next[matrixKind][row][column] = value;
    setWeights(next);
  }

  function updateOutputWeight(row: number, column: number, value: number) {
    if (!weights || !Number.isFinite(value)) return;
    const next = cloneWeights(weights);
    next.output[row][column] = value;
    setWeights(next);
  }

  function resetOutputWeights() {
    if (!weights || !defaultWeights) return;
    const next = cloneWeights(weights);
    next.output = defaultWeights.output.map((row) => [...row]);
    setWeights(next);
  }

  function updateNormalization(kind: "gamma" | "beta", dimension: number, value: number) {
    if (!normalization || !Number.isFinite(value)) return;
    setNormalization({ ...normalization, [kind]: normalization[kind].map((item, index) => index === dimension ? value : item) });
  }

  function resetNormalization() {
    if (!normalization || !defaultNormalization) return;
    setNormalization({
      gamma: [...defaultNormalization.gamma],
      beta: [...defaultNormalization.beta],
      epsilon: normalization.epsilon,
    });
  }

  function updateFeedForwardVector(field: "input_bias" | "output_bias", index: number, value: number) {
    if (!feedForward || !Number.isFinite(value)) return;
    const next = cloneFeedForward(feedForward);
    next[field][index] = value;
    setFeedForward(next);
  }

  function updateFeedForwardMatrix(field: "input_weights" | "output_weights", row: number, column: number, value: number) {
    if (!feedForward || !Number.isFinite(value)) return;
    const next = cloneFeedForward(feedForward);
    next[field][row][column] = value;
    setFeedForward(next);
  }

  function updateFeedForwardNormalization(field: "gamma" | "beta", index: number, value: number) {
    if (!feedForward || !Number.isFinite(value)) return;
    const next = cloneFeedForward(feedForward);
    next.normalization[field][index] = value;
    setFeedForward(next);
  }

  function resetFeedForward() {
    if (defaultFeedForward) setFeedForward(cloneFeedForward(defaultFeedForward));
  }

  const selectedProjection = result?.projections.find((item) => item.position === selectedPosition) ?? result?.projections[0];
  const selectedBreakdown = selectedProjection?.breakdown[projectionKind][outputDimension];
  const selectedAttentionCalculation = result?.attention.calculations.find(
    (item) => item.query_position === selectedPosition && item.key_position === selectedKeyPosition,
  );
  const selectedContextCalculation = result?.attention.context_calculations
    .find((item) => item.query_position === selectedPosition)?.dimensions[contextDimension];
  const attentionMatrix = result?.attention[attentionView] ?? [];
  const selectedHeadResult = result?.multi_head_attention.heads[selectedHead];
  const selectedOutputCalculation = result?.multi_head_attention.output_calculations
    .find((item) => item.position === selectedPosition)?.dimensions[multiHeadOutputDimension];
  const selectedNormalizationCalculation = result?.attention_sublayer.calculations
    .find((item) => item.position === selectedPosition);
  const selectedNormalizationTerm = selectedNormalizationCalculation?.normalization_terms[normalizationDimension];
  const normalizedMean = selectedNormalizationCalculation ? vectorMean(selectedNormalizationCalculation.normalized_vector) : 0;
  const normalizedVariance = selectedNormalizationCalculation ? vectorVariance(selectedNormalizationCalculation.normalized_vector) : 0;
  const selectedFeedForwardCalculation = result?.feed_forward_sublayer.calculations
    .find((item) => item.position === selectedPosition);
  const selectedHiddenCalculation = selectedFeedForwardCalculation?.hidden_calculations[selectedHiddenNeuron];
  const selectedFeedForwardOutputCalculation = selectedFeedForwardCalculation?.output_calculations[feedForwardOutputDimension];

  return (
    <div className="toy-lab">
      <section className="hero">
        <p className="eyebrow">Transformer Attention Visualizer</p>
        <h1>See how tokens prepare to communicate.</h1>
        <p className="hero-copy">Follow text through token and position embeddings, then inspect the projections that create query, key, and value vectors.</p>
      </section>

      <section className="workspace" aria-labelledby="workspace-title">
        <div className="workspace-heading">
          <div><p className="step-label">Phases 1–2</p><h2 id="workspace-title">Input & embedding pipeline</h2></div>
          <span className="phase-badge">Complete</span>
        </div>

        <form onSubmit={handleSubmit}>
          <label htmlFor="input-text">Text to inspect</label>
          <div className="input-row">
            <input id="input-text" value={text} onChange={(event) => setText(event.target.value)} maxLength={500} placeholder="Try: I love AI!" />
            <button type="submit" disabled={isLoading}>{isLoading ? "Inspecting…" : "Start inspection"}</button>
          </div>
          <p className="input-hint">Words and punctuation are tokenized separately · Maximum 500 characters</p>
        </form>

        {error && <p className="error-message">{error}</p>}

        {result && (
          <div className="inspection" aria-live="polite">
            <div className="metrics">
              <div><span>Tokens</span><strong>{result.token_count}</strong></div>
              <div><span>Vocabulary</span><strong>{result.vocabulary_size}</strong></div>
              <div><span>Dimensions</span><strong>{result.embedding_dimension}D</strong></div>
              <div><span>Characters</span><strong>{result.character_count}</strong></div>
            </div>

            <div className="token-strip" aria-label="Token sequence">
              {result.tokens.map((item) => <div className="token-chip" key={`${item.position}-${item.token}`}><span>pos {item.position}</span><strong>{item.token}</strong><small>ID {item.token_id}{item.normalized === "<UNK>" ? " · UNK" : ""}</small></div>)}
            </div>

            <div className="formula"><span>combined input</span><b>=</b><span>token embedding</span><b>+</b><span>position embedding</span></div>
            <div className="heatmap-heading">
              <div><p className="step-label">Embedding heatmap</p><h3>{embeddingLabels[embeddingKind]} embedding</h3></div>
              <div className="switcher" role="group" aria-label="Choose embedding heatmap">
                {(Object.keys(embeddingLabels) as EmbeddingKind[]).map((kind) => <button type="button" className={kind === embeddingKind ? "selected" : ""} onClick={() => setEmbeddingKind(kind)} key={kind}>{embeddingLabels[kind]}</button>)}
              </div>
            </div>
            <div className="heatmap-scroll">
              <div className="heatmap" style={{ gridTemplateColumns: `minmax(92px, 1fr) repeat(${result.embedding_dimension}, minmax(78px, 1fr))` }}>
                <div className="heat-label">Token</div>
                {Array.from({ length: result.embedding_dimension }, (_, index) => <div className="heat-label" key={index}>d{index}</div>)}
                {result.tokens.map((item) => <div className="heat-row" key={`${embeddingKind}-${item.position}`}><div className="heat-token"><strong>{item.token}</strong><span>pos {item.position} · id {item.token_id}</span></div>{item[embeddingKind].map((value, index) => <div className="heat-cell" style={{ background: heatColor(value) }} key={index}>{value.toFixed(3)}</div>)}</div>)}
              </div>
            </div>
          </div>
        )}
      </section>

      {result && weights && (
        <section className="workspace projection-workspace" aria-labelledby="projection-title">
          <div className="workspace-heading">
            <div><p className="step-label">03 — Q, K and V projections</p><h2 id="projection-title">Project each token into three roles</h2></div>
            <span className="phase-badge">Active</span>
          </div>

          <div className="role-cards">
            <p><strong>Q asks:</strong> “What information is this token looking for?”</p>
            <p><strong>K describes:</strong> “What information does this token contain?”</p>
            <p><strong>V carries:</strong> “What information will be passed forward?”</p>
          </div>

          <div className="matrix-panel">
            <div className="matrix-toolbar">
              <div className="switcher" role="group" aria-label="Choose weight matrix">
                {(Object.keys(projectionLabels) as ProjectionKind[]).map((kind) => <button type="button" className={kind === matrixKind ? "selected" : ""} onClick={() => setMatrixKind(kind)} key={kind}>{projectionSymbols[kind]}</button>)}
              </div>
              <strong>{projectionLabels[matrixKind]}</strong>
            </div>
            <div className="matrix-grid" aria-label={projectionLabels[matrixKind]}>
              {weights[matrixKind].map((row, rowIndex) => row.map((value, columnIndex) => (
                <label className="matrix-cell" key={`${rowIndex}-${columnIndex}`}><span>w{rowIndex}{columnIndex}</span><input type="number" step="0.01" value={value} onChange={(event) => updateWeight(rowIndex, columnIndex, event.currentTarget.valueAsNumber)} /></label>
              )))}
            </div>
            <div className="matrix-actions">
              <button type="button" disabled={isLoading} onClick={() => void inspect(weights, normalization ?? undefined, feedForward ?? undefined)}>Recalculate QKV</button>
              <button type="button" className="secondary-button" disabled={!defaultWeights || isLoading} onClick={() => defaultWeights && setWeights(cloneWeights(defaultWeights))}>Reset default weights</button>
            </div>
          </div>

          <div className="projection-inspector">
            <div className="selector-block"><span>Inspect token</span><div className="token-selector">{result.projections.map((item) => <button type="button" className={item.position === selectedPosition ? "selected" : ""} onClick={() => setSelectedPosition(item.position)} key={item.position}>{item.token}</button>)}</div></div>
            {selectedProjection && (
              <>
                <div className="projection-flow"><span>Combined embedding X</span><b>→</b><span>multiply by WQ, WK and WV</span><b>→</b><span>Q, K and V vectors</span></div>
                <div className="vector-stack">
                  {(["query", "key", "value"] as ProjectionKind[]).map((kind) => <div className="vector-row" key={kind}><strong>{projectionSymbols[kind]}</strong>{selectedProjection[kind].map((value, index) => <span className="vector-cell" style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div>)}
                </div>

                <div className="breakdown-panel">
                  <div className="breakdown-controls">
                    <div className="switcher">{(["query", "key", "value"] as ProjectionKind[]).map((kind) => <button type="button" className={kind === projectionKind ? "selected" : ""} onClick={() => setProjectionKind(kind)} key={kind}>{projectionSymbols[kind]}</button>)}</div>
                    <div className="dimension-selector">{[0, 1, 2, 3].map((dimension) => <button type="button" className={dimension === outputDimension ? "selected" : ""} onClick={() => setOutputDimension(dimension)} key={dimension}>d{dimension}</button>)}</div>
                  </div>
                  {selectedBreakdown && <div className="equation"><strong>{projectionSymbols[projectionKind].toLowerCase()}<sub>{outputDimension}</sub> =</strong><div>{selectedBreakdown.terms.map((term, index) => <span key={term.input_dimension}>{index > 0 && <b> + </b>} (x<sub>{term.input_dimension}</sub> {term.input_value.toFixed(4)} × w<sub>{term.input_dimension}{outputDimension}</sub> {term.weight_value.toFixed(4)})</span>)}</div><strong>= {selectedBreakdown.result.toFixed(4)}</strong></div>}
                </div>
              </>
            )}
          </div>
        </section>
      )}

      {result && (
        <section className="workspace attention-workspace" aria-labelledby="attention-title">
          <div className="workspace-heading">
            <div><p className="step-label">04 — Scaled dot-product attention</p><h2 id="attention-title">Decide what each token can use</h2></div>
            <span className="phase-badge">Active</span>
          </div>

          <div className="attention-pipeline" aria-label="Attention calculation pipeline">
            {["Q × Kᵀ", `Scale by √dₖ (${result.attention.scale_factor.toFixed(1)})`, "Apply causal mask", "Softmax", "Attention weights", "Multiply by V", "Context vectors"].map((step, index) => <span key={step}>{index > 0 && <b>→</b>}{step}</span>)}
          </div>

          <p className="mask-note"><strong>Causal mask:</strong> A decoder Transformer cannot look at future tokens while predicting the next token.</p>

          <div className="attention-controls">
            <div className="switcher attention-view-switcher" role="group" aria-label="Choose attention matrix">
              {(Object.keys(attentionViewLabels) as AttentionView[]).map((view) => <button type="button" className={view === attentionView ? "selected" : ""} onClick={() => setAttentionView(view)} key={view}>{attentionViewLabels[view]}</button>)}
            </div>
            <div className="selector-block"><span>Query token</span><div className="token-selector">{result.tokens.map((token) => <button type="button" className={token.position === selectedPosition ? "selected" : ""} onClick={() => setSelectedPosition(token.position)} key={token.position}>{token.token}</button>)}</div></div>
          </div>

          <div className="attention-table-scroll">
            <div className="attention-table" style={{ gridTemplateColumns: `minmax(90px, 1fr) repeat(${result.token_count}, minmax(86px, 1fr)) 95px` }}>
              <div className="attention-axis">Query ↓ / Key →</div>
              {result.tokens.map((token) => <div className="attention-axis" key={token.position}>{token.token}<small>k{token.position}</small></div>)}
              <div className="attention-axis">Check</div>
              {attentionMatrix.map((row, queryPosition) => <div className="attention-table-row" key={queryPosition}>
                <div className={`attention-axis row-label ${queryPosition === selectedPosition ? "selected" : ""}`}>{result.tokens[queryPosition].token}<small>q{queryPosition}</small></div>
                {row.map((value, keyPosition) => {
                  const masked = result.attention.masked_scores[queryPosition][keyPosition] === null;
                  const displayValue = attentionView === "masked_scores" && masked ? null : value;
                  return <button type="button" className={`attention-cell ${queryPosition === selectedPosition ? "selected-row" : ""} ${selectedAttentionCalculation?.query_position === queryPosition && selectedAttentionCalculation?.key_position === keyPosition ? "selected-cell" : ""} ${displayValue === null ? "masked" : ""}`} style={displayValue === null ? undefined : { background: attentionView === "attention_weights" ? `rgba(116, 235, 178, ${0.08 + Math.min(Math.abs(displayValue), 1) * 0.75})` : heatColor(displayValue) }} onClick={() => { setSelectedPosition(queryPosition); setSelectedKeyPosition(keyPosition); }} key={keyPosition}>{displayValue === null ? "Masked" : Number(displayValue).toFixed(4)}</button>;
                })}
                <div className={`row-sum ${Math.abs(result.attention.row_sums[queryPosition] - 1) <= 0.0001 ? "valid" : ""}`}>Sum = {result.attention.row_sums[queryPosition].toFixed(4)}</div>
              </div>)}
            </div>
          </div>

          <div className="attention-detail-grid">
            <div className="connection-panel">
              <p className="step-label">Attention connections</p>
              <h3>{result.tokens[selectedPosition]?.token} attends backward</h3>
              <svg className="connection-graph" viewBox="0 0 800 230" role="img" aria-label={`Attention connections for ${result.tokens[selectedPosition]?.token}`}>
                {result.tokens.map((token, keyPosition) => {
                  const weight = result.attention.attention_weights[selectedPosition]?.[keyPosition] ?? 0;
                  if (!result.attention.causal_mask[selectedPosition]?.[keyPosition]) return null;
                  const keyX = 70 + keyPosition * (660 / Math.max(result.token_count - 1, 1));
                  return <g key={keyPosition}><line x1="400" y1="55" x2={keyX} y2="175" stroke="#74ebb2" strokeWidth={1 + weight * 12} opacity={0.22 + weight * 0.78} /><text x={(400 + keyX) / 2} y={108 + Math.abs(400 - keyX) * 0.035} fill="#d8e5e0" fontSize="14" textAnchor="middle">{weight.toFixed(4)}</text><circle cx={keyX} cy="185" r="25" fill="#10221e" stroke="#4da97d" /><text x={keyX} y="190" fill="#e9f2ef" fontSize="14" textAnchor="middle">{token.token}</text></g>;
                })}
                <circle cx="400" cy="42" r="29" fill="#173c30" stroke="#74ebb2" /><text x="400" y="47" fill="#f4faf7" fontSize="15" textAnchor="middle">{result.tokens[selectedPosition]?.token}</text>
              </svg>
            </div>

            <div className="score-inspector">
              <p className="step-label">Selected score calculation</p>
              {selectedAttentionCalculation && <>
                <h3>score({selectedAttentionCalculation.query_token}, {selectedAttentionCalculation.key_token})</h3>
                <div className="score-equation">{selectedAttentionCalculation.products.map((product, index) => <span key={product.dimension}>{index > 0 && <b> + </b>}(q<sub>{product.dimension}</sub> {product.query_value.toFixed(4)} × k<sub>{product.dimension}</sub> {product.key_value.toFixed(4)})</span>)}</div>
                <dl className="calculation-summary"><div><dt>Raw score</dt><dd>{selectedAttentionCalculation.raw_score.toFixed(4)}</dd></div><div><dt>Scaled ÷ √{result.attention.key_dimension}</dt><dd>{selectedAttentionCalculation.scaled_score.toFixed(4)}</dd></div><div><dt>Causal mask</dt><dd>{selectedAttentionCalculation.causally_masked ? "Masked" : "Allowed"}</dd></div><div><dt>Softmax weight</dt><dd>{selectedAttentionCalculation.attention_weight.toFixed(4)}</dd></div></dl>
              </>}
            </div>
          </div>

          <div className="context-panel">
            <div className="heatmap-heading"><div><p className="step-label">Context vectors</p><h3>Z = attention weights × V</h3></div><div className="dimension-selector">{[0, 1, 2, 3].map((dimension) => <button type="button" className={dimension === contextDimension ? "selected" : ""} onClick={() => setContextDimension(dimension)} key={dimension}>z{dimension}</button>)}</div></div>
            <div className="vector-stack">{result.attention.context_vectors.map((vector, queryPosition) => <div className="vector-row context-row" key={queryPosition}><strong>{result.tokens[queryPosition].token}</strong>{vector.map((value, dimension) => <span className="vector-cell" style={{ background: heatColor(value) }} key={dimension}>{value.toFixed(4)}</span>)}</div>)}</div>
            {selectedContextCalculation && <div className="context-equation"><strong>z<sub>{contextDimension}</sub> for {result.tokens[selectedPosition].token} =</strong>{selectedContextCalculation.terms.map((term, index) => <span key={term.key_position}>{index > 0 && <b> + </b>}({term.attention_weight.toFixed(4)} × V[{term.key_token}]<sub>{contextDimension}</sub> {term.value.toFixed(4)})</span>)}<strong>= {selectedContextCalculation.result.toFixed(4)}</strong></div>}
          </div>
        </section>
      )}

      {result && weights && (
        <section className="workspace multi-head-workspace" aria-labelledby="multi-head-title">
          <div className="workspace-heading">
            <div><p className="step-label">05 — Multi-head attention</p><h2 id="multi-head-title">Read through multiple perspectives</h2></div>
            <span className="phase-badge">Active</span>
          </div>

          <div className="multi-head-pipeline">
            {["Full Q, K and V", "Split dimensions", "Head 1 attention", "Head 2 attention", "Concatenate contexts", "Multiply by WO", "Multi-head output"].map((step, index) => <span key={step}>{index > 0 && <b>→</b>}{step}</span>)}
          </div>
          <p className="head-explanation">Different attention heads can learn different relationships between tokens. A high attention weight shows where a head is reading information from; it does not, by itself, completely explain the model’s decision.</p>

          <div className="multi-head-controls">
            <div className="selector-block"><span>Query token</span><div className="token-selector">{result.tokens.map((token) => <button type="button" className={token.position === selectedPosition ? "selected" : ""} onClick={() => setSelectedPosition(token.position)} key={token.position}>{token.token}</button>)}</div></div>
            <div className="switcher" role="group" aria-label="Select attention head">{result.multi_head_attention.heads.map((head) => <button type="button" className={head.head_index === selectedHead ? "selected" : ""} onClick={() => setSelectedHead(head.head_index)} key={head.head_index}>Head {head.head_index + 1}</button>)}</div>
          </div>

          {selectedHeadResult && <div className="dimension-split">
            {result.multi_head_attention.heads.map((head) => <div className={head.head_index === selectedHead ? "head-split selected" : "head-split"} key={head.head_index}>
              <div><span>Head {head.head_index + 1}</span><strong>{head.dimension_indices.map((dimension) => `d${dimension}`).join(", ")}</strong></div>
              {(["Q", "K", "V"] as const).map((symbol) => {
                const vectors = symbol === "Q" ? head.query_vectors : symbol === "K" ? head.key_vectors : head.value_vectors;
                return <p key={symbol}><b>{symbol}</b>{vectors[selectedPosition]?.map((value, index) => <span key={index}>{value.toFixed(4)}</span>)}</p>;
              })}
            </div>)}
          </div>}

          <div className="head-heatmaps">
            {result.multi_head_attention.heads.map((head) => <div className="head-heatmap-card" key={head.head_index}>
              <div className="head-card-title"><strong>Head {head.head_index + 1}</strong><span>d{head.dimension_indices[0]}, d{head.dimension_indices[1]} · scale √2</span></div>
              <div className="head-table" style={{ gridTemplateColumns: `80px repeat(${result.token_count}, minmax(70px, 1fr)) 80px` }}>
                <div className="attention-axis">Q ↓ / K →</div>{result.tokens.map((token) => <div className="attention-axis" key={token.position}>{token.token}</div>)}<div className="attention-axis">Sum</div>
                {head.attention_weights.map((row, queryPosition) => <div className="attention-table-row" key={queryPosition}><div className={`attention-axis row-label ${queryPosition === selectedPosition ? "selected" : ""}`}>{result.tokens[queryPosition].token}</div>{row.map((value, keyPosition) => head.masked_scores[queryPosition][keyPosition] === null ? <div className="head-attention-cell masked" key={keyPosition}>Masked</div> : <button type="button" className={`head-attention-cell ${queryPosition === selectedPosition ? "selected-row" : ""}`} style={{ background: `rgba(116, 235, 178, ${0.08 + value * 0.75})` }} onClick={() => { setSelectedHead(head.head_index); setSelectedPosition(queryPosition); setSelectedKeyPosition(keyPosition); }} key={keyPosition}>{value.toFixed(4)}</button>)}<div className="row-sum valid">{head.row_sums[queryPosition].toFixed(4)}</div></div>)}
              </div>
            </div>)}
          </div>

          {selectedHeadResult && <div className="head-detail-grid">
            <div className="connection-panel">
              <p className="step-label">Head {selectedHead + 1} connections</p><h3>{result.tokens[selectedPosition]?.token} reads from</h3>
              <svg className="connection-graph" viewBox="0 0 800 230" role="img" aria-label={`Head ${selectedHead + 1} connections`}>
                {result.tokens.map((token, keyPosition) => { const weight = selectedHeadResult.attention_weights[selectedPosition]?.[keyPosition] ?? 0; if (!selectedHeadResult.causal_mask[selectedPosition]?.[keyPosition]) return null; const keyX = 70 + keyPosition * (660 / Math.max(result.token_count - 1, 1)); return <g key={keyPosition}><line x1="400" y1="55" x2={keyX} y2="175" stroke="#74ebb2" strokeWidth={1 + weight * 12} opacity={0.22 + weight * 0.78} /><text x={(400 + keyX) / 2} y={110 + Math.abs(400 - keyX) * 0.03} fill="#d8e5e0" fontSize="14" textAnchor="middle">{weight.toFixed(4)}</text><circle cx={keyX} cy="185" r="25" fill="#10221e" stroke="#4da97d" /><text x={keyX} y="190" fill="#e9f2ef" fontSize="14" textAnchor="middle">{token.token}</text></g>; })}
                <circle cx="400" cy="42" r="29" fill="#173c30" stroke="#74ebb2" /><text x="400" y="47" fill="#f4faf7" fontSize="15" textAnchor="middle">{result.tokens[selectedPosition]?.token}</text>
              </svg>
            </div>
            <div className="head-context-panel"><p className="step-label">Selected token contexts</p><h3>{result.tokens[selectedPosition]?.token}</h3>{result.multi_head_attention.heads.map((head) => <div className="compact-vector" key={head.head_index}><strong>Head {head.head_index + 1}</strong>{head.context_vectors[selectedPosition]?.map((value, index) => <span style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div>)}<div className="compact-vector concatenated"><strong>Concatenated</strong>{result.multi_head_attention.concatenated_contexts[selectedPosition]?.map((value, index) => <span style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div></div>
          </div>}

          <div className="output-projection-panel">
            <div className="heatmap-heading"><div><p className="step-label">Output projection</p><h3>WO — Output projection weights</h3></div><div className="matrix-actions"><button type="button" disabled={isLoading} onClick={() => void inspect(weights, normalization ?? undefined, feedForward ?? undefined)}>Recalculate output</button><button type="button" className="secondary-button" disabled={!defaultWeights || isLoading} onClick={resetOutputWeights}>Reset WO</button></div></div>
            <div className="output-layout">
              <div className="matrix-grid">{weights.output.map((row, rowIndex) => row.map((value, columnIndex) => <label className="matrix-cell" key={`${rowIndex}-${columnIndex}`}><span>wo{rowIndex}{columnIndex}</span><input type="number" step="0.01" value={value} onChange={(event) => updateOutputWeight(rowIndex, columnIndex, event.currentTarget.valueAsNumber)} /></label>))}</div>
              <div className="output-flow"><div><span>Concatenated context</span><div className="mini-vector">{result.multi_head_attention.concatenated_contexts[selectedPosition]?.map((value, index) => <b key={index}>{value.toFixed(4)}</b>)}</div></div><strong>× WO =</strong><div><span>Multi-head output</span><div className="mini-vector">{result.multi_head_attention.projected_outputs[selectedPosition]?.map((value, index) => <b style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</b>)}</div></div></div>
            </div>
            <div className="breakdown-controls output-breakdown-controls"><span>Inspect output dimension</span><div className="dimension-selector">{[0, 1, 2, 3].map((dimension) => <button type="button" className={dimension === multiHeadOutputDimension ? "selected" : ""} onClick={() => setMultiHeadOutputDimension(dimension)} key={dimension}>d{dimension}</button>)}</div></div>
            {selectedOutputCalculation && <div className="context-equation"><strong>output<sub>{multiHeadOutputDimension}</sub> =</strong>{selectedOutputCalculation.terms.map((term, index) => <span key={term.input_dimension}>{index > 0 && <b> + </b>}(c<sub>{term.input_dimension}</sub> {term.concatenated_input_value.toFixed(4)} × wo<sub>{term.input_dimension}{multiHeadOutputDimension}</sub> {term.weight_value.toFixed(4)})</span>)}<strong>= {selectedOutputCalculation.result.toFixed(4)}</strong></div>}
          </div>
        </section>
      )}

      {result && weights && normalization && selectedNormalizationCalculation && (
        <section className="workspace normalization-workspace" aria-labelledby="normalization-title">
          <div className="workspace-heading">
            <div><p className="step-label">06 — Residual connection and layer normalization</p><h2 id="normalization-title">Preserve, combine, and stabilize</h2></div>
            <span className="phase-badge">Active</span>
          </div>

          <div className="architecture-label"><span>Educational post-normalization</span><strong>LayerNorm(X + Attention(X))</strong></div>
          <div className="normalization-flow">
            {["Original combined embedding X", "+ Multi-head attention output", "= Residual vector", "Center using mean", "Scale using variance", "Apply gamma and beta", "Layer-normalized output"].map((step, index) => <span key={step}>{index > 0 && <b>→</b>}{step}</span>)}
          </div>

          <div className="norm-explanations">
            <p><strong>Residual connection</strong>The residual connection preserves the token’s original information while adding the information collected by attention.</p>
            <p><strong>Layer normalization</strong>Layer normalization keeps the four features of each token on a stable scale. Gamma and beta are learnable parameters that allow the model to adjust the normalized values.</p>
          </div>

          <div className="selector-block norm-token-selector"><span>Inspect token</span><div className="token-selector">{result.tokens.map((token) => <button type="button" className={token.position === selectedPosition ? "selected" : ""} onClick={() => setSelectedPosition(token.position)} key={token.position}>{token.token}</button>)}</div></div>

          <div className="residual-panel">
            <p className="step-label">First residual connection</p>
            <div className="norm-vector-table">
              <div className="norm-vector-header">Vector</div>{[0, 1, 2, 3].map((dimension) => <div className="norm-vector-header" key={dimension}>d{dimension}</div>)}
              {[
                { label: "Original input X", values: selectedNormalizationCalculation.input_vector, tone: "input" },
                { label: "Attention output", values: selectedNormalizationCalculation.attention_output, tone: "attention" },
                { label: "Residual result", values: selectedNormalizationCalculation.residual_vector, tone: "residual" },
              ].map((row) => <div className="norm-vector-row" key={row.label}><strong>{row.label}</strong>{row.values.map((value, index) => <span className={`norm-cell ${row.tone}`} style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div>)}
            </div>
            <div className="residual-equations">{selectedNormalizationCalculation.residual_terms.map((term) => <div key={term.dimension}><strong>d{term.dimension}</strong><span>{term.input_value.toFixed(4)}</span><b>+</b><span>{term.attention_value.toFixed(4)}</span><b>=</b><span>{term.residual_value.toFixed(4)}</span></div>)}</div>
          </div>

          <div className="normalization-panel">
            <div className="normalization-summary">
              <div><span>Mean</span><strong>{selectedNormalizationCalculation.mean.toFixed(4)}</strong></div>
              <div><span>Population variance</span><strong>{selectedNormalizationCalculation.variance.toFixed(4)}</strong></div>
              <div><span>Epsilon</span><strong>{selectedNormalizationCalculation.epsilon.toFixed(5)}</strong></div>
              <div><span>Std. deviation</span><strong>{selectedNormalizationCalculation.standard_deviation.toFixed(4)}</strong></div>
            </div>

            <div className="norm-stages">
              {[
                { label: "Residual", values: selectedNormalizationCalculation.residual_vector },
                { label: "Centered", values: selectedNormalizationCalculation.normalization_terms.map((term) => term.centered_value) },
                { label: "Normalized", values: selectedNormalizationCalculation.normalized_vector },
                { label: "LayerNorm output", values: selectedNormalizationCalculation.layer_norm_output },
              ].map((stage) => <div key={stage.label}><strong>{stage.label}</strong><div>{stage.values.map((value, index) => <span style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div></div>)}
            </div>

            <div className="norm-validation"><div className={Math.abs(normalizedMean) <= 0.0001 ? "valid" : ""}><span>Normalized mean</span><strong>{normalizedMean.toFixed(4)}</strong><small>Expected ≈ 0</small></div><div className={Math.abs(normalizedVariance - 1) <= 0.001 ? "valid" : ""}><span>Normalized variance</span><strong>{normalizedVariance.toFixed(4)}</strong><small>Expected ≈ 1</small></div></div>
          </div>

          <div className="affine-panel">
            <div className="heatmap-heading"><div><p className="step-label">Learnable affine parameters</p><h3>Gamma and beta</h3></div><div className="matrix-actions"><button type="button" disabled={isLoading} onClick={() => void inspect(weights, normalization, feedForward ?? undefined)}>Recalculate normalization</button><button type="button" className="secondary-button" disabled={!defaultNormalization || isLoading} onClick={resetNormalization}>Reset gamma and beta</button></div></div>
            <div className="affine-editors">
              {(["gamma", "beta"] as const).map((kind) => <div key={kind}><strong>{kind === "gamma" ? "Gamma — scale" : "Beta — shift"}</strong><div>{normalization[kind].map((value, dimension) => <label className="matrix-cell" key={dimension}><span>{kind === "gamma" ? "γ" : "β"}{dimension}</span><input type="number" step="0.1" value={value} onChange={(event) => updateNormalization(kind, dimension, event.currentTarget.valueAsNumber)} /></label>)}</div></div>)}
            </div>

            <div className="breakdown-controls norm-breakdown-controls"><span>Inspect dimension</span><div className="dimension-selector">{[0, 1, 2, 3].map((dimension) => <button type="button" className={dimension === normalizationDimension ? "selected" : ""} onClick={() => setNormalizationDimension(dimension)} key={dimension}>d{dimension}</button>)}</div></div>
            {selectedNormalizationTerm && <div className="normalization-equation"><div><strong>normalized<sub>{normalizationDimension}</sub> =</strong><span>({selectedNormalizationTerm.residual_value.toFixed(4)} − {selectedNormalizationTerm.mean.toFixed(4)}) / √({selectedNormalizationCalculation.variance.toFixed(4)} + {selectedNormalizationCalculation.epsilon.toFixed(5)})</span><strong>= {selectedNormalizationTerm.normalized_value.toFixed(4)}</strong></div><div><strong>output<sub>{normalizationDimension}</sub> =</strong><span>{selectedNormalizationTerm.gamma.toFixed(4)} × {selectedNormalizationTerm.normalized_value.toFixed(4)} + {selectedNormalizationTerm.beta.toFixed(4)}</span><strong>= {selectedNormalizationTerm.output_value.toFixed(4)}</strong></div></div>}
          </div>
        </section>
      )}

      {result && weights && normalization && feedForward && selectedFeedForwardCalculation && (
        <section className="workspace feed-forward-workspace" aria-labelledby="feed-forward-title">
          <div className="workspace-heading"><div><p className="step-label">07 — Feed-forward network and complete Transformer block</p><h2 id="feed-forward-title">Transform each token independently</h2></div><span className="phase-badge">Active</span></div>
          <div className="block-formula"><span>TransformerBlock(X) =</span><strong>LayerNorm2(LayerNorm1(X + MultiHeadAttention(X)) + FFN(LayerNorm1(X + MultiHeadAttention(X))))</strong></div>
          <div className="ffn-flow">{["Phase 6 normalized token", "W1 projection", "Add b1", "ReLU", "W2 projection", "Add b2", "FFN output", "Second residual", "LayerNorm2", "Block output"].map((step, index) => <span key={step}>{index > 0 && <b>→</b>}{step}</span>)}</div>
          <div className="ffn-explanations"><p><strong>Feed-forward network</strong>The feed-forward network processes each token independently after attention has mixed information between tokens.</p><p><strong>ReLU</strong>ReLU keeps positive values and changes negative values to zero.</p><p><strong>Second residual connection</strong>The second residual connection preserves the attention-stage representation while adding the feed-forward transformation.</p></div>

          <div className="selector-block"><span>Inspect token</span><div className="token-selector">{result.tokens.map((token) => <button type="button" className={token.position === selectedPosition ? "selected" : ""} onClick={() => setSelectedPosition(token.position)} key={token.position}>{token.token}</button>)}</div></div>
          <div className="ffn-input"><span>Phase 6 LayerNorm output</span><div>{selectedFeedForwardCalculation.input_vector.map((value, index) => <b style={{ background: heatColor(value) }} key={index}>d{index}<small>{value.toFixed(4)}</small></b>)}</div></div>

          <div className="hidden-panel">
            <div className="heatmap-heading"><div><p className="step-label">Eight-neuron hidden layer</p><h3>W1 × input + b1 → ReLU</h3></div></div>
            <div className="neuron-grid">{selectedFeedForwardCalculation.hidden_calculations.map((neuron) => <button type="button" className={`${neuron.is_active ? "active-neuron" : "inactive-neuron"} ${neuron.hidden_neuron === selectedHiddenNeuron ? "selected" : ""}`} onClick={() => setSelectedHiddenNeuron(neuron.hidden_neuron)} key={neuron.hidden_neuron}><strong>h{neuron.hidden_neuron}</strong><span>Pre {neuron.pre_activation.toFixed(4)}</span><span>ReLU {neuron.activated.toFixed(4)}</span><small>{neuron.is_active ? "Active" : "Inactive"}</small></button>)}</div>
            <p className="toy-note">Active and inactive labels show the ReLU calculation only; these toy neurons do not have assigned human-interpretable meanings.</p>
            {selectedHiddenCalculation && <div className="ffn-equation"><strong>hidden<sub>{selectedHiddenNeuron}</sub> =</strong><div>{selectedHiddenCalculation.terms.map((term, index) => <span key={term.input_dimension}>{index > 0 && <b> + </b>}(y<sub>{term.input_dimension}</sub> {term.input_value.toFixed(4)} × w<sub>{term.input_dimension}{selectedHiddenNeuron}</sub> {term.weight_value.toFixed(4)})</span>)}<b> + bias {selectedHiddenCalculation.bias.toFixed(4)}</b></div><strong>= {selectedHiddenCalculation.pre_activation.toFixed(4)}</strong><strong>ReLU = {selectedHiddenCalculation.activated.toFixed(4)} · {selectedHiddenCalculation.is_active ? "Active" : "Inactive"}</strong></div>}
            <div className="wide-vector"><span>Pre-activation</span>{selectedFeedForwardCalculation.pre_activation_vector.map((value, index) => <b style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</b>)}</div>
            <div className="wide-vector"><span>After ReLU</span>{selectedFeedForwardCalculation.activated_vector.map((value, index) => <b className={value > 0 ? "active-value" : "inactive-value"} key={index}>{value.toFixed(4)}</b>)}</div>
          </div>

          <div className="ffn-output-panel">
            <div className="breakdown-controls"><span>Inspect FFN output dimension</span><div className="dimension-selector">{[0, 1, 2, 3].map((dimension) => <button type="button" className={dimension === feedForwardOutputDimension ? "selected" : ""} onClick={() => setFeedForwardOutputDimension(dimension)} key={dimension}>d{dimension}</button>)}</div></div>
            {selectedFeedForwardOutputCalculation && <div className="ffn-equation"><strong>ffn_output<sub>{feedForwardOutputDimension}</sub> =</strong><div>{selectedFeedForwardOutputCalculation.terms.map((term, index) => <span key={term.input_dimension}>{index > 0 && <b> + </b>}(h<sub>{term.input_dimension}</sub> {term.input_value.toFixed(4)} × w2<sub>{term.input_dimension}{feedForwardOutputDimension}</sub> {term.weight_value.toFixed(4)})</span>)}<b> + bias {selectedFeedForwardOutputCalculation.bias.toFixed(4)}</b></div><strong>= {selectedFeedForwardOutputCalculation.output.toFixed(4)}</strong></div>}
          </div>

          <div className="ffn-parameters">
            <div className="heatmap-heading"><div><p className="step-label">Advanced controls</p><h3>Feed-forward parameters</h3></div><div className="matrix-actions"><button type="button" disabled={isLoading} onClick={() => void inspect(weights, normalization, feedForward)}>Recalculate FFN</button><button type="button" className="secondary-button" disabled={!defaultFeedForward || isLoading} onClick={resetFeedForward}>Reset FFN parameters</button></div></div>
            <div className="parameter-tabs">{(["input_weights", "input_bias", "output_weights", "output_bias", "normalization"] as FeedForwardEditor[]).map((tab) => <button type="button" className={tab === feedForwardEditor ? "selected" : ""} onClick={() => setFeedForwardEditor(tab)} key={tab}>{({ input_weights: "W1 · 4×8", input_bias: "b1 · 8", output_weights: "W2 · 8×4", output_bias: "b2 · 4", normalization: "LayerNorm2" })[tab]}</button>)}</div>
            <div className="parameter-editor">
              {(feedForwardEditor === "input_weights" || feedForwardEditor === "output_weights") && <div className={`ffn-matrix ${feedForwardEditor === "input_weights" ? "w1" : "w2"}`}>{feedForward[feedForwardEditor].map((row, rowIndex) => row.map((value, columnIndex) => <label className="matrix-cell" key={`${rowIndex}-${columnIndex}`}><span>w{rowIndex}{columnIndex}</span><input type="number" step="0.01" value={value} onChange={(event) => updateFeedForwardMatrix(feedForwardEditor, rowIndex, columnIndex, event.currentTarget.valueAsNumber)} /></label>))}</div>}
              {(feedForwardEditor === "input_bias" || feedForwardEditor === "output_bias") && <div className="ffn-bias">{feedForward[feedForwardEditor].map((value, index) => <label className="matrix-cell" key={index}><span>b{index}</span><input type="number" step="0.01" value={value} onChange={(event) => updateFeedForwardVector(feedForwardEditor, index, event.currentTarget.valueAsNumber)} /></label>)}</div>}
              {feedForwardEditor === "normalization" && <div className="affine-editors">{(["gamma", "beta"] as const).map((field) => <div key={field}><strong>{field}</strong><div>{feedForward.normalization[field].map((value, index) => <label className="matrix-cell" key={index}><span>{field === "gamma" ? "γ" : "β"}{index}</span><input type="number" step="0.1" value={value} onChange={(event) => updateFeedForwardNormalization(field, index, event.currentTarget.valueAsNumber)} /></label>)}</div></div>)}</div>}
            </div>
          </div>

          <div className="block-output-panel">
            <p className="step-label">Second residual and LayerNorm2</p>
            <div className="residual-equations">{selectedFeedForwardCalculation.residual_terms.map((term) => <div key={term.dimension}><strong>d{term.dimension}</strong><span>{term.input_value.toFixed(4)}</span><b>+</b><span>{term.feed_forward_value.toFixed(4)}</span><b>=</b><span>{term.residual_value.toFixed(4)}</span></div>)}</div>
            <div className="normalization-summary"><div><span>Mean</span><strong>{selectedFeedForwardCalculation.mean.toFixed(4)}</strong></div><div><span>Population variance</span><strong>{selectedFeedForwardCalculation.variance.toFixed(4)}</strong></div><div><span>Std. deviation</span><strong>{selectedFeedForwardCalculation.standard_deviation.toFixed(4)}</strong></div><div><span>Epsilon</span><strong>{feedForward.normalization.epsilon.toFixed(5)}</strong></div></div>
            <div className="norm-stages block-stages">{[
              { label: "Phase 6 input", values: selectedFeedForwardCalculation.input_vector }, { label: "FFN output", values: selectedFeedForwardCalculation.feed_forward_output }, { label: "Residual2", values: selectedFeedForwardCalculation.residual_vector }, { label: "Normalized", values: selectedFeedForwardCalculation.normalized_vector }, { label: "Gamma", values: feedForward.normalization.gamma }, { label: "Beta", values: feedForward.normalization.beta }, { label: "Transformer block", values: selectedFeedForwardCalculation.transformer_block_output },
            ].map((stage) => <div key={stage.label}><strong>{stage.label}</strong><div>{stage.values.map((value, index) => <span style={{ background: heatColor(value) }} key={index}>{value.toFixed(4)}</span>)}</div></div>)}</div>
          </div>
        </section>
      )}

      <section className="pipeline" aria-label="Project pipeline">
        {["Input", "Embedding", "Q · K · V", "Attention", "Multi-head", "Residual + Norm", "Feed-forward block", "Prediction"].map((item, index) => <div className={index <= 6 ? "pipeline-step complete" : "pipeline-step next-mode"} key={item}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item}</strong></div>)}
      </section>

      <section className="prediction-handoff" aria-labelledby="prediction-handoff-title">
        <div>
          <p className="step-label">Next: trained model</p>
          <h2 id="prediction-handoff-title">Prediction is in its own lab.</h2>
          <p>The transparent Toy Math Lab finishes at the feed-forward block. To predict after <strong>I love</strong>, load the saved trained checkpoint in the Predict mode.</p>
        </div>
        <div className="prediction-handoff-actions">
          <a className="secondary-link-button" href="#train">Train model</a>
          <a className="primary-link-button" href="#predict">Open prediction →</a>
        </div>
      </section>
    </div>
  );
}
