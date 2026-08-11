import { FormEvent, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { JsonObject, JsonValue, ModelStatus, PredictionItem, TinyInspectResponse } from "../types/api";
import Heatmap from "./common/Heatmap";
import ProbabilityBars from "./common/ProbabilityBars";
import StatusNotice from "./common/StatusNotice";

type Stage = "embeddings" | "qkv" | "attention" | "context" | "residual" | "ffn" | "output";
type Path = Array<string | number>;
type InspectSelection = {
  text: string;
  layer: number;
  head: number;
  queryToken: number;
  keyToken: number;
  hiddenDimension: number;
  topK: number;
};

const STAGES: Array<{ id: Stage; label: string }> = [
  { id: "embeddings", label: "Embeddings" },
  { id: "qkv", label: "Q · K · V" },
  { id: "attention", label: "Attention" },
  { id: "context", label: "Head context" },
  { id: "residual", label: "Residual stream" },
  { id: "ffn", label: "Feed-forward" },
  { id: "output", label: "Final output" },
];

function object(value: JsonValue | undefined): JsonObject | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value as JsonObject : undefined;
}

function at(root: JsonValue, path: Path): JsonValue | undefined {
  let current: JsonValue | undefined = root;
  for (const part of path) {
    if (typeof part === "number") {
      if (!Array.isArray(current)) return undefined;
      current = current[part];
    } else {
      const record = object(current);
      if (!record) return undefined;
      current = record[part];
    }
  }
  return current;
}

function first(root: JsonValue, paths: Path[]): JsonValue | undefined {
  for (const path of paths) {
    const value = at(root, path);
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function numberVector(value: JsonValue | undefined): number[] {
  return Array.isArray(value) && value.every((item) => typeof item === "number" && Number.isFinite(item)) ? value as number[] : [];
}

function stringVector(value: JsonValue | undefined): string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string") ? value as string[] : [];
}

function numberMatrix(value: JsonValue | undefined): number[][] {
  if (!Array.isArray(value)) return [];
  const rows = value.map(numberVector);
  return rows.length && rows.every((row) => row.length > 0) ? rows : [];
}

function nullableMatrix(value: JsonValue | undefined): Array<Array<number | null>> {
  if (!Array.isArray(value)) return [];
  const rows = value.map((row) => Array.isArray(row) && row.every((item) => item === null || (typeof item === "number" && Number.isFinite(item))) ? row as Array<number | null> : []);
  return rows.length && rows.every((row) => row.length > 0) ? rows : [];
}

function maskMatrix(value: JsonValue | undefined): Array<Array<number | null>> {
  if (!Array.isArray(value)) return [];
  const rows = value.map((row) => Array.isArray(row) ? row.map((item) => item === true || item === 1 ? 1 : item === false || item === 0 || item === null ? null : null) : []);
  return rows.length && rows.every((row) => row.length > 0) ? rows : [];
}

function selectedMatrix(value: JsonValue | undefined, index: number): number[][] {
  const direct = numberMatrix(value);
  if (direct.length) return direct;
  if (!Array.isArray(value)) return [];
  return numberMatrix(value[index]) || [];
}

function matrixFrom(root: JsonValue, paths: Path[], nestedIndex?: number): number[][] {
  const value = first(root, paths);
  return nestedIndex === undefined ? numberMatrix(value) : selectedMatrix(value, nestedIndex);
}

function predictionsFrom(value: JsonValue | undefined): PredictionItem[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const record = object(item);
    if (!record) return [];
    const token = record.token;
    const tokenId = record.token_id;
    const logit = record.logit;
    const probability = record.probability;
    return typeof token === "string" && typeof tokenId === "number" && typeof logit === "number" && typeof probability === "number"
      ? [{ token, token_id: tokenId, logit, probability }]
      : [];
  });
}

function tensorShape(matrix: Array<Array<number | null>>): string {
  return matrix.length ? `${matrix.length} × ${Math.max(...matrix.map((row) => row.length))}` : "not returned";
}

type InspectorData = {
  tokens: string[];
  tokenIds: number[];
  tokenEmbeddings: number[][];
  positionEmbeddings: number[][];
  combinedEmbeddings: number[][];
  query: number[][];
  key: number[][];
  value: number[][];
  rawScores: Array<Array<number | null>>;
  causalMask: Array<Array<number | null>>;
  attentionProbabilities: number[][];
  headContexts: number[][];
  concatenatedAttention: number[][];
  attentionResidual: number[][];
  normalizedFfnInput: number[][];
  ffnPreActivations: number[][];
  geluActivations: number[][];
  blockOutputs: number[][];
  finalHiddenStates: number[][];
  predictions: PredictionItem[];
  selectedHiddenDimension: number;
  selectedHiddenValues: number[];
};

function adaptTrace(response: TinyInspectResponse, layer: number, head: number): InspectorData {
  const root: JsonValue = response;
  const layerRoot = first(root, [["trace", "layers", layer], ["layers", layer], ["layer_trace"]]);
  const localRoot = layerRoot ?? root;
  const local = (paths: Path[]) => first(localRoot, paths) ?? first(root, paths);
  return {
    tokens: stringVector(first(root, [["tokens"], ["input", "tokens"], ["trace", "tokens"]])),
    tokenIds: numberVector(first(root, [["token_ids"], ["input", "token_ids"], ["trace", "token_ids"]])),
    tokenEmbeddings: matrixFrom(root, [["token_embeddings"], ["embeddings", "token"], ["trace", "token_embeddings"]]),
    positionEmbeddings: matrixFrom(root, [["position_embeddings"], ["embeddings", "position"], ["trace", "position_embeddings"]]),
    combinedEmbeddings: matrixFrom(root, [["combined_embeddings"], ["embeddings", "combined"], ["trace", "combined_embeddings"]]),
    query: selectedMatrix(local([["queries"], ["query"], ["q"], ["query_vectors"], ["attention", "query"]]), head),
    key: selectedMatrix(local([["keys"], ["key"], ["k"], ["key_vectors"], ["attention", "key"]]), head),
    value: selectedMatrix(local([["values"], ["value"], ["v"], ["value_vectors"], ["attention", "value"]]), head),
    rawScores: nullableMatrix(selectedMatrix(local([["raw_attention_scores"], ["raw_scores"], ["attention", "raw_scores"]]), head)),
    causalMask: maskMatrix(local([["causal_mask"], ["attention", "causal_mask"]])),
    attentionProbabilities: selectedMatrix(local([["attention_probabilities"], ["attention_weights"], ["attention", "probabilities"]]), head),
    headContexts: selectedMatrix(local([["head_context_vectors"], ["context_vectors"], ["attention", "head_contexts"]]), head),
    concatenatedAttention: numberMatrix(local([["concatenated_attention_output"], ["concatenated_attention_outputs"], ["concatenated_attention"], ["attention", "concatenated_output"]])),
    attentionResidual: numberMatrix(local([["attention_residual_output"], ["attention_residual_outputs"], ["attention_residual"], ["residual_outputs"]])),
    normalizedFfnInput: numberMatrix(local([["normalized_feed_forward_input"], ["normalized_feed_forward_inputs"], ["normalized_ffn_input"], ["feed_forward", "normalized_input"]])),
    ffnPreActivations: numberMatrix(local([["feed_forward_pre_activations"], ["ffn_pre_activations"], ["feed_forward", "pre_activations"]])),
    geluActivations: numberMatrix(local([["gelu_activations"], ["feed_forward", "gelu_activations"]])),
    blockOutputs: numberMatrix(local([["block_output"], ["block_outputs"], ["output"], ["residual_outputs", "block"]])),
    finalHiddenStates: matrixFrom(root, [["final_hidden_states"], ["trace", "final_hidden_states"]]),
    predictions: predictionsFrom(first(root, [["predictions"], ["top_predictions"], ["trace", "predictions"]])),
    selectedHiddenDimension: typeof at(root, ["selected_hidden_values", "dimension"]) === "number" ? at(root, ["selected_hidden_values", "dimension"]) as number : 0,
    selectedHiddenValues: numberVector(at(root, ["selected_hidden_values", "final_hidden_states"])),
  };
}

function ConnectionGraph({ tokens, weights, query }: { tokens: string[]; weights: number[][]; query: number }) {
  const row = weights[query] ?? [];
  if (!tokens.length || !row.length) return <div className="empty-state compact">Attention connections are not available.</div>;
  const span = Math.max(tokens.length - 1, 1);
  return <svg className="trained-connection-graph" viewBox="0 0 800 260" role="img" aria-label={`Attention pattern from ${tokens[query] ?? `token ${query}`}`}><title>Selected head attention pattern. Line thickness and opacity represent attention weight.</title>{tokens.map((token, index) => { const weight = row[index] ?? 0; const x = 65 + index * 670 / span; return <g key={`${token}-${index}`}><line x1="400" y1="65" x2={x} y2="190" stroke="#74ebb2" strokeWidth={1 + Math.max(0, weight) * 14} opacity={0.12 + Math.max(0, weight) * 0.82} /><text x={(400 + x) / 2} y={125} fill="#d9e6e1" fontSize="13" textAnchor="middle">{weight.toFixed(4)}</text><circle cx={x} cy="205" r="26" fill="#10221e" stroke="#4da97d" /><text x={x} y="210" fill="#eff8f4" fontSize="13" textAnchor="middle">{token}</text></g>; })}<circle cx="400" cy="48" r="31" fill="#173c30" stroke="#74ebb2" strokeWidth="2" /><text x="400" y="53" fill="#f4faf7" fontSize="14" textAnchor="middle">{tokens[query]}</text></svg>;
}

export default function TrainedInspector() {
  const [text, setText] = useState("I love");
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [queryToken, setQueryToken] = useState(0);
  const [keyToken, setKeyToken] = useState(0);
  const [hiddenDimension, setHiddenDimension] = useState(0);
  const [topK, setTopK] = useState(5);
  const [stage, setStage] = useState<Stage>("attention");
  const [response, setResponse] = useState<TinyInspectResponse | null>(null);
  const [appliedSelection, setAppliedSelection] = useState<InspectSelection | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api.modelStatus().then((next) => active && setModelStatus(next)).catch((requestError: unknown) => active && setError(requestError instanceof Error ? requestError.message : "Unable to read model status."));
    return () => { active = false; };
  }, []);

  const data = useMemo(() => response ? adaptTrace(response, response.selection.layer, response.selection.head) : null, [response]);
  const layerCount = modelStatus?.model_config?.number_of_layers ?? 2;
  const headCount = modelStatus?.model_config?.number_of_heads ?? 4;
  const appliedQuery = response?.selection.query_token ?? 0;
  const appliedKey = response?.selection.key_token ?? 0;
  const selectionIsPending = appliedSelection !== null && (
    text.trim() !== appliedSelection.text
    || layer !== appliedSelection.layer
    || head !== appliedSelection.head
    || queryToken !== appliedSelection.queryToken
    || keyToken !== appliedSelection.keyToken
    || hiddenDimension !== appliedSelection.hiddenDimension
    || topK !== appliedSelection.topK
  );

  async function inspect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      const next = await api.inspectModel({ text, layer, head, query_token: queryToken, key_token: keyToken, hidden_dimension: hiddenDimension, top_k: topK });
      setResponse(next);
      setAppliedSelection({
        text: next.input_text,
        layer: next.selection.layer,
        head: next.selection.head,
        queryToken: next.selection.query_token,
        keyToken: next.selection.key_token,
        hiddenDimension: next.selection.hidden_dimension,
        topK,
      });
      setLayer(next.selection.layer);
      setHead(next.selection.head);
      setQueryToken(next.selection.query_token);
      setKeyToken(next.selection.key_token);
      setHiddenDimension(next.selection.hidden_dimension);
      const tokens = stringVector(first(next, [["tokens"], ["input", "tokens"], ["trace", "tokens"]]));
      if (!tokens.length) setError("The backend returned a trace without tokens.");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The model trace could not be created.");
    } finally {
      setIsLoading(false);
    }
  }

  const labels = data?.tokens ?? [];
  const selectedProbability = data?.attentionProbabilities[appliedQuery]?.[appliedKey];
  const traceShapes = response ? Object.entries(response.shapes ?? {}).filter((entry): entry is [string, number[]] => Array.isArray(entry[1]) && entry[1].every((value) => typeof value === "number")) : [];
  const selectedCalculation = response?.selected_attention_calculation;

  return (
    <div className="mode-page inspector-page">
      <header className="mode-hero"><p className="eyebrow">Actual trained tensors</p><h1>Inspect where a tiny model reads.</h1><p>Trace one layer and head at a time. Attention patterns show where a head reads information from; they are not a complete explanation of a prediction.</p></header>
      <section className="lab-panel inspector-controls" aria-labelledby="inspect-controls-title">
        <div className="panel-heading"><div><p className="step-label">Trace selection</p><h2 id="inspect-controls-title">Choose a model slice</h2></div><span className="phase-badge">Modern pre-norm</span></div>
        <form onSubmit={inspect}>
          <label className="prompt-field">Prompt<input value={text} maxLength={500} onChange={(event) => setText(event.currentTarget.value)} /></label>
          <div className="selector-grid"><label>Layer<select value={layer} onChange={(event) => setLayer(Number(event.currentTarget.value))}>{Array.from({ length: layerCount }, (_, index) => <option value={index} key={index}>Layer {index}</option>)}</select></label><label>Head<select value={head} onChange={(event) => setHead(Number(event.currentTarget.value))}>{Array.from({ length: headCount }, (_, index) => <option value={index} key={index}>Head {index}</option>)}</select></label><label>Query token<select value={queryToken} onChange={(event) => setQueryToken(Number(event.currentTarget.value))}>{labels.length ? labels.map((token, index) => <option value={index} key={`${token}-${index}`}>{index}: {token}</option>) : <option value="0">0 — inspect first</option>}</select></label><label>Key token<select value={keyToken} onChange={(event) => setKeyToken(Number(event.currentTarget.value))}>{labels.length ? labels.map((token, index) => <option value={index} key={`${token}-${index}`}>{index}: {token}</option>) : <option value="0">0 — inspect first</option>}</select></label><label>Hidden dimension<input type="number" min="0" max={Math.max(0, (modelStatus?.model_config?.d_model ?? 32) - 1)} value={hiddenDimension} onChange={(event) => setHiddenDimension(event.currentTarget.valueAsNumber)} /></label><label>Top vocabulary results<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(event.currentTarget.valueAsNumber)} /></label></div>
          <button type="submit" disabled={isLoading || !text.trim()}>{isLoading ? "Tracing…" : "Inspect trained model"}</button>
        </form>
        {selectionIsPending && <StatusNotice kind="warning">The controls have changed. Select “Inspect trained model” to apply them; the visualizations below still show the last completed backend trace.</StatusNotice>}
        {error && <StatusNotice kind="error">{error}</StatusNotice>}
      </section>

      {data ? <>
        <section className="tensor-overview" aria-label="Trace tensor shapes"><div><span>Tokens</span><strong>{data.tokens.length}</strong></div><div><span>Q shape shown</span><strong>{tensorShape(data.query)}</strong></div><div><span>Attention shape shown</span><strong>{tensorShape(data.attentionProbabilities)}</strong></div><div><span>Final hidden shown</span><strong>{tensorShape(data.finalHiddenStates)}</strong></div></section>
        <details className="tensor-shape-details"><summary>All backend tensor shapes</summary><dl>{traceShapes.map(([name, shape]) => <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{shape.join(" × ")}</dd></div>)}</dl></details>
        <nav className="stage-tabs" aria-label="Calculation stage">{STAGES.map((item) => <button type="button" className={stage === item.id ? "selected" : ""} aria-current={stage === item.id ? "page" : undefined} onClick={() => setStage(item.id)} key={item.id}>{item.label}</button>)}</nav>

        <section className="lab-panel trace-stage" aria-live="polite">
          {stage === "embeddings" && <><div className="panel-heading"><div><p className="step-label">Input representation</p><h2>Token + learned position embeddings</h2></div><span className="shape-badge">sequence × d_model</span></div><div className="stacked-heatmaps"><Heatmap title="Token embedding heatmap" values={data.tokenEmbeddings} rowLabels={labels} /><Heatmap title="Learned position embedding heatmap" values={data.positionEmbeddings} rowLabels={labels} /><Heatmap title="Combined embedding heatmap" values={data.combinedEmbeddings} rowLabels={labels} /></div></>}
          {stage === "qkv" && <><div className="panel-heading"><div><p className="step-label">Layer {response?.selection.layer} · Head {response?.selection.head}</p><h2>Query, key and value projections</h2></div><span className="shape-badge">sequence × d_head</span></div><div className="stacked-heatmaps three"><Heatmap title="Q — what each token requests" values={data.query} rowLabels={labels} /><Heatmap title="K — what each token advertises" values={data.key} rowLabels={labels} /><Heatmap title="V — information carried forward" values={data.value} rowLabels={labels} /></div></>}
          {stage === "attention" && <><div className="panel-heading"><div><p className="step-label">Causal self-attention</p><h2>Scores, mask and reading pattern</h2></div><span className="shape-badge">{tensorShape(data.attentionProbabilities)}</span></div><div className="attention-trace-grid"><Heatmap title="Raw attention scores" values={data.rawScores} rowLabels={labels} columnLabels={labels} selectedCell={[appliedQuery, appliedKey]} onCellSelect={(row, column) => { setQueryToken(row); setKeyToken(column); }} /><Heatmap title="Causal mask — Allowed or Masked" values={data.causalMask} rowLabels={labels} columnLabels={labels} format={() => "Allowed"} /><Heatmap title="Attention probabilities" values={data.attentionProbabilities} rowLabels={labels} columnLabels={labels} selectedCell={[appliedQuery, appliedKey]} onCellSelect={(row, column) => { setQueryToken(row); setKeyToken(column); }} /></div><div className="selected-readout"><span>Applied backend pattern</span><strong>{labels[appliedQuery]} → {labels[appliedKey]}</strong><b>{typeof selectedProbability === "number" ? selectedProbability.toFixed(6) : "not returned"}</b></div>{selectedCalculation && <dl className="attention-calculation-summary"><div><dt>Raw Q · K</dt><dd>{selectedCalculation.raw_score.toFixed(6)}</dd></div><div><dt>Scale √d_head</dt><dd>{selectedCalculation.scale_factor.toFixed(6)}</dd></div><div><dt>Scaled score</dt><dd>{selectedCalculation.scaled_score.toFixed(6)}</dd></div><div><dt>Mask</dt><dd>{selectedCalculation.causally_masked ? "Future — blocked" : "Allowed"}</dd></div><div><dt>Probability</dt><dd>{selectedCalculation.attention_probability.toFixed(6)}</dd></div></dl>}<ConnectionGraph tokens={labels} weights={data.attentionProbabilities} query={appliedQuery} /><StatusNotice>{response?.attention_note ?? "Attention patterns show where this head reads; they do not completely explain the prediction."}</StatusNotice></>}
          {stage === "context" && <><div className="panel-heading"><div><p className="step-label">Weighted value mixture</p><h2>Head and concatenated context vectors</h2></div></div><div className="stacked-heatmaps"><Heatmap title={`Head ${response?.selection.head} context vectors`} values={data.headContexts} rowLabels={labels} /><Heatmap title="Concatenated attention output" values={data.concatenatedAttention} rowLabels={labels} /></div></>}
          {stage === "residual" && <><div className="panel-heading"><div><p className="step-label">Information highway</p><h2>Residual stream vectors</h2></div><span className="shape-badge">pre-normalization block</span></div><div className="stacked-heatmaps"><Heatmap title="After attention residual" values={data.attentionResidual} rowLabels={labels} /><Heatmap title="Layer-normalized FFN input" values={data.normalizedFfnInput} rowLabels={labels} /><Heatmap title="Block output" values={data.blockOutputs} rowLabels={labels} /></div></>}
          {stage === "ffn" && <><div className="panel-heading"><div><p className="step-label">Per-token transformation</p><h2>Feed-forward activations</h2></div><span className="shape-badge">sequence × d_ff</span></div><div className="stacked-heatmaps"><Heatmap title="FFN pre-activation" values={data.ffnPreActivations} rowLabels={labels} /><Heatmap title="GELU activation" values={data.geluActivations} rowLabels={labels} /><Heatmap title="Block output" values={data.blockOutputs} rowLabels={labels} /></div></>}
          {stage === "output" && <><div className="panel-heading"><div><p className="step-label">Language-model head</p><h2>Final hidden state and next-token distribution</h2></div><span className="shape-badge">trained vocabulary</span></div><Heatmap title="Final hidden-state heatmap" values={data.finalHiddenStates} rowLabels={labels} /><div className="selected-hidden-row"><span>Selected hidden dimension d{data.selectedHiddenDimension}</span>{data.selectedHiddenValues.map((value, index) => <b key={index}>{labels[index] ?? `t${index}`}<small>{value.toFixed(4)}</small></b>)}</div><div className="top-output-probabilities"><h3>Top next-token model probabilities</h3><ProbabilityBars predictions={data.predictions} /><div className={`sum-check ${Math.abs((response?.probability_sum ?? 0) - 1) <= 0.0001 ? "valid" : "invalid"}`}><strong>Full vocabulary probability sum</strong><span>{(response?.probability_sum ?? 0).toFixed(6)}</span></div></div></>}
        </section>
      </> : <section className="empty-state inspector-empty"><strong>No trained-model trace yet.</strong><span>Load a checkpoint, choose a layer and head, then inspect “I love”.</span></section>}
    </div>
  );
}
