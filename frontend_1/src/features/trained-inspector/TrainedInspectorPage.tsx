import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../../api/endpoints";
import { useWorkspace } from "../../app/workspaceContext";
import Heatmap, { type HeatmapSelection } from "../../components/common/Heatmap";
import ModelLoadGate from "../../components/common/ModelLoadGate";
import ProbabilityBars from "../../components/common/ProbabilityBars";
import StatusNotice from "../../components/common/StatusNotice";
import type { ModelStatus, TinyInspectResponse } from "../../types/api";

type InspectorStage = "embeddings" | "qkv" | "attention" | "context" | "residual" | "ffn" | "output";
type AttentionTraceView = "raw" | "scaled" | "mask" | "probabilities";

const STAGES: ReadonlyArray<{ id: InspectorStage; label: string }> = [
  { id: "embeddings", label: "Embeddings" },
  { id: "qkv", label: "Q, K and V" },
  { id: "attention", label: "Attention" },
  { id: "context", label: "Head context" },
  { id: "residual", label: "Residual stream" },
  { id: "ffn", label: "Feed-forward" },
  { id: "output", label: "Final hidden state" },
];

function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function tokenLabel(token: string): string {
  return token.length === 0 ? "<empty>" : token.replaceAll(" ", "␠").replaceAll("\n", "↵");
}

function windowed(matrix: readonly (readonly number[])[], start: number, size: number): number[][] {
  return matrix.map((row) => row.slice(start, start + size));
}

function dimensionLabels(start: number, count: number, prefix = "d"): string[] {
  return Array.from({ length: count }, (_, index) => `${prefix}${start + index}`);
}

function maskFuture(allowed: readonly (readonly boolean[])[]): boolean[][] {
  return allowed.map((row) => row.map((value) => !value));
}

function matrixWidth(matrix: readonly (readonly number[])[]): number {
  return matrix[0]?.length ?? 0;
}

export default function TrainedInspectorPage() {
  const workspace = useWorkspace();
  const observedRunNonce = useRef(workspace.runNonce);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [response, setResponse] = useState<TinyInspectResponse | null>(null);
  const [prompt, setPrompt] = useState(workspace.prompt);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [queryToken, setQueryToken] = useState(0);
  const [keyToken, setKeyToken] = useState(0);
  const [stage, setStage] = useState<InspectorStage>("attention");
  const [attentionView, setAttentionView] = useState<AttentionTraceView>("probabilities");
  const [dimensionStart, setDimensionStart] = useState(0);
  const [windowSize, setWindowSize] = useState(8);
  const [topK, setTopK] = useState(5);
  const [loadingModel, setLoadingModel] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState("");
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.modelStatus({ signal: controller.signal })
      .then(setModelStatus)
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Unable to read the local model status."));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => () => request.current?.abort(), []);

  const modelConfig = response?.model_config ?? modelStatus?.model_config;
  const layerCount = modelConfig?.number_of_layers ?? 2;
  const headCount = modelConfig?.number_of_heads ?? 4;
  const modelDimension = modelConfig?.d_model ?? 32;
  const feedForwardDimension = modelConfig?.feed_forward_dimension ?? 64;
  const headDimension = Math.max(1, Math.floor(modelDimension / headCount));
  const tokenCount = response?.tokens.length ?? 1;
  const stageDimension = stage === "qkv" || stage === "context"
    ? headDimension
    : stage === "ffn"
      ? feedForwardDimension
      : modelDimension;
  const maximumStart = Math.max(0, stageDimension - 1);
  const effectiveDimensionStart = Math.min(dimensionStart, maximumStart);

  async function loadCheckpoint() {
    setLoadingModel(true);
    setError("");
    try {
      setModelStatus(await api.loadModel());
    } catch (requestError: unknown) {
      setError(messageFrom(requestError, "The best checkpoint could not be loaded."));
    } finally {
      setLoadingModel(false);
    }
  }

  const inspect = useCallback(async (nextQuery = queryToken, nextKey = keyToken, nextPrompt = prompt) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setInspecting(true);
    setError("");
    try {
      const next = await api.inspectModel({
        text: nextPrompt,
        layer,
        head,
        query_token: nextQuery,
        key_token: nextKey,
        hidden_dimension: Math.min(effectiveDimensionStart, modelDimension - 1),
        top_k: topK,
      }, { signal: controller.signal });
      setResponse(next);
      setLayer(next.selection.layer);
      setHead(next.selection.head);
      setQueryToken(next.selection.query_token);
      setKeyToken(next.selection.key_token);
    } catch (requestError: unknown) {
      if (!controller.signal.aborted) setError(messageFrom(requestError, "Model inspection failed."));
    } finally {
      if (request.current === controller) {
        request.current = null;
        setInspecting(false);
      }
    }
  }, [effectiveDimensionStart, head, keyToken, layer, modelDimension, prompt, queryToken, topK]);

  function selectAttentionCell(selection: HeatmapSelection) {
    setQueryToken(selection.row);
    setKeyToken(selection.column);
    void inspect(selection.row, selection.column);
  }

  useEffect(() => {
    if (observedRunNonce.current === workspace.runNonce) return;
    observedRunNonce.current = workspace.runNonce;
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      setPrompt(workspace.prompt);
      if (modelStatus?.loaded) void inspect(queryToken, keyToken, workspace.prompt);
    });
    return () => { active = false; };
  }, [inspect, keyToken, modelStatus?.loaded, queryToken, workspace.prompt, workspace.runNonce]);

  const rowLabels = response?.tokens.map(tokenLabel) ?? [];
  const dimensionCount = Math.min(windowSize, Math.max(0, stageDimension - effectiveDimensionStart));
  const columns = dimensionLabels(effectiveDimensionStart, dimensionCount, stage === "ffn" ? "h" : "d");

  return (
    <main className="feature-page trained-inspector-page" aria-labelledby="trained-inspector-title">
      <header className="feature-heading">
        <p className="eyebrow">Real local trace</p>
        <h1 id="trained-inspector-title">Inspect one layer and head precisely.</h1>
        <p>Large tensors stay manageable through a labelled dimension window; exact selected values remain available on demand.</p>
      </header>

      <ModelLoadGate
        loaded={modelStatus?.loaded ?? false}
        loading={loadingModel}
        checkpointAvailable={modelStatus?.checkpoint_available}
        device={modelStatus?.device}
        onLoad={() => void loadCheckpoint()}
      />
      {error ? <StatusNotice kind="error">{error}</StatusNotice> : null}

      <section className="workspace-section inspector-controls" aria-labelledby="trace-controls-title">
        <div className="section-heading"><div><p className="section-kicker">Trace request</p><h2 id="trace-controls-title">Choose a narrow view</h2></div></div>
        <label className="prompt-control">Text<textarea rows={2} maxLength={500} value={prompt} onChange={(event) => { setPrompt(event.currentTarget.value); workspace.setPrompt(event.currentTarget.value); }} /></label>
        <div className="control-grid">
          <label>Layer<select value={layer} onChange={(event) => setLayer(Number(event.currentTarget.value))}>{Array.from({ length: layerCount }, (_, index) => <option value={index} key={index}>Layer {index}</option>)}</select></label>
          <label>Head<select value={head} onChange={(event) => setHead(Number(event.currentTarget.value))}>{Array.from({ length: headCount }, (_, index) => <option value={index} key={index}>Head {index}</option>)}</select></label>
          <label>Query token<select value={Math.min(queryToken, tokenCount - 1)} onChange={(event) => setQueryToken(Number(event.currentTarget.value))}>{Array.from({ length: tokenCount }, (_, index) => <option value={index} key={index}>{response ? `${index}: ${tokenLabel(response.tokens[index] ?? "")}` : index}</option>)}</select></label>
          <label>Key token<select value={Math.min(keyToken, tokenCount - 1)} onChange={(event) => setKeyToken(Number(event.currentTarget.value))}>{Array.from({ length: tokenCount }, (_, index) => <option value={index} key={index}>{response ? `${index}: ${tokenLabel(response.tokens[index] ?? "")}` : index}</option>)}</select></label>
          <label>Top predictions<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.currentTarget.value))} /></label>
          <button type="button" disabled={!modelStatus?.loaded || inspecting || !prompt.trim()} onClick={() => void inspect()}>{inspecting ? "Inspecting…" : "Run inspection"}</button>
        </div>
      </section>

      <section className="workspace-section trace-workspace" aria-labelledby="trace-stage-title">
        <div className="stage-toolbar">
          <div className="segmented-control" role="tablist" aria-label="Trace stage">
            {STAGES.map((item) => (
              <button type="button" role="tab" aria-selected={stage === item.id} key={item.id} onClick={() => setStage(item.id)}>{item.label}</button>
            ))}
          </div>
          <div className="dimension-window-controls">
            <label>First dimension<input type="number" min="0" max={maximumStart} value={effectiveDimensionStart} onChange={(event) => setDimensionStart(Number(event.currentTarget.value))} /></label>
            <label>Window<select value={windowSize} onChange={(event) => setWindowSize(Number(event.currentTarget.value))}><option value="4">4</option><option value="8">8</option><option value="16">16</option></select></label>
          </div>
        </div>
        <div className="section-heading"><div><p className="section-kicker">Selected trace</p><h2 id="trace-stage-title">{STAGES.find((item) => item.id === stage)?.label}</h2></div><span>{effectiveDimensionStart}–{Math.max(effectiveDimensionStart, effectiveDimensionStart + dimensionCount - 1)} of {stageDimension}</span></div>
        {response ? (
          <TraceStage
            response={response}
            stage={stage}
            dimensionStart={effectiveDimensionStart}
            dimensionCount={dimensionCount}
            columns={columns}
            rowLabels={rowLabels}
            selection={{ row: response.selection.query_token, column: response.selection.key_token }}
            onSelectAttention={selectAttentionCell}
            attentionView={attentionView}
            onAttentionViewChange={setAttentionView}
          />
        ) : <p className="empty-state">Load the local checkpoint and request layer 0, head 0 to begin.</p>}
      </section>
    </main>
  );
}

type TraceStageProps = {
  response: TinyInspectResponse;
  stage: InspectorStage;
  dimensionStart: number;
  dimensionCount: number;
  columns: string[];
  rowLabels: string[];
  selection: HeatmapSelection;
  onSelectAttention: (selection: HeatmapSelection) => void;
  attentionView: AttentionTraceView;
  onAttentionViewChange: (view: AttentionTraceView) => void;
};

function TraceStage({ response, stage, dimensionStart, dimensionCount, columns, rowLabels, selection, onSelectAttention, attentionView, onAttentionViewChange }: TraceStageProps) {
  const trace = response.layer_trace;
  if (stage === "embeddings") {
    return <div className="trace-stack">
      <Heatmap title="Token embeddings" values={windowed(response.token_embeddings, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Position embeddings" values={windowed(response.position_embeddings, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Combined embeddings" values={windowed(response.combined_embeddings, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    </div>;
  }

  if (stage === "qkv") {
    return <div className="trace-stack">
      <p className="flow-equation">Normalized hidden state → selected head’s Q, K and V projections</p>
      <Heatmap title="Query vectors" values={windowed(trace.query, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Key vectors" values={windowed(trace.key, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Value vectors" values={windowed(trace.value, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    </div>;
  }

  if (stage === "attention") {
    const futureMask = maskFuture(trace.causal_mask);
    const rowSum = trace.attention_probabilities[selection.row]?.reduce((total, value) => total + value, 0) ?? 0;
    const calculation = response.selected_attention_calculation;
    return <div className="trace-stack">
      <div className="segmented-control attention-substage-selector" role="tablist" aria-label="Attention calculation stage">
        <button type="button" role="tab" aria-selected={attentionView === "raw"} onClick={() => onAttentionViewChange("raw")}>Raw QKᵀ scores</button>
        <button type="button" role="tab" aria-selected={attentionView === "scaled"} onClick={() => onAttentionViewChange("scaled")}>Scaled scores</button>
        <button type="button" role="tab" aria-selected={attentionView === "mask"} onClick={() => onAttentionViewChange("mask")}>Causal mask</button>
        <button type="button" role="tab" aria-selected={attentionView === "probabilities"} onClick={() => onAttentionViewChange("probabilities")}>Probabilities</button>
      </div>
      {attentionView === "raw" ? <>
        <p className="flow-equation">Raw score = QKᵀ. Future-token scores may exist here, but the causal mask blocks them before softmax.</p>
        <Heatmap title="Raw attention scores (QKᵀ)" values={trace.raw_attention_scores} rowLabels={rowLabels} columnLabels={rowLabels} selected={selection} onSelect={onSelectAttention} />
      </> : null}
      {attentionView === "scaled" ? <>
        <p className="flow-equation">Scaled score = raw score ÷ √dₕ. These are the real pre-mask scaled values returned by the backend.</p>
        <Heatmap title="Scaled attention scores" values={trace.scaled_attention_scores} rowLabels={rowLabels} columnLabels={rowLabels} selected={selection} onSelect={onSelectAttention} />
      </> : null}
      {attentionView === "mask" ? <>
        <p className="flow-equation">Backend mask semantics: Visible means the key is at or before the query. Masked means a future key is blocked and receives probability 0.</p>
        <CausalMaskTable allowed={trace.causal_mask} rowLabels={rowLabels} columnLabels={rowLabels} selected={selection} onSelect={onSelectAttention} />
      </> : null}
      {attentionView === "probabilities" ? <>
        <p className="flow-equation">Softmax is applied after the causal mask. Every masked future position has weight 0.</p>
        <Heatmap title="Attention probabilities after causal mask and softmax" values={trace.attention_probabilities} rowLabels={rowLabels} columnLabels={rowLabels} mask={futureMask} selected={selection} onSelect={onSelectAttention} />
        <p className="row-sum-validation">Selected row sum: <strong>{rowSum.toFixed(6)}</strong> · {Math.abs(rowSum - 1) <= 0.0001 ? "approximately 1 ✓" : "check normalization"}</p>
      </> : null}
      <AttentionConnections response={response} />
      <div className="calculation-inspector">
        <h3>{tokenLabel(calculation.query_token)} → {tokenLabel(calculation.key_token)}</h3>
        <p>{calculation.products.map((product) => product.toFixed(4)).join(" + ")} = {calculation.raw_score.toFixed(4)}</p>
        <dl>
          <div><dt>Raw score</dt><dd>{calculation.raw_score.toFixed(4)}</dd></div>
          <div><dt>Scale</dt><dd>√{calculation.scale_factor.toFixed(4)}</dd></div>
          <div><dt>Scaled score</dt><dd>{calculation.scaled_score.toFixed(4)}</dd></div>
          <div><dt>Probability</dt><dd>{calculation.causally_masked ? "Masked · 0.0000" : calculation.attention_probability.toFixed(4)}</dd></div>
        </dl>
      </div>
      <StatusNotice kind="warning">Attention shows where this head reads information from; it is not a complete explanation of the prediction.</StatusNotice>
    </div>;
  }

  if (stage === "context") {
    return <div className="trace-stack">
      <Heatmap title="Selected-head context" values={windowed(trace.head_context_vectors, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Concatenated heads" values={windowed(trace.concatenated_attention_output, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Output projection" values={windowed(trace.projected_attention_output, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    </div>;
  }

  if (stage === "residual") {
    return <div className="trace-stack">
      <Heatmap title="Projected attention contribution" values={windowed(trace.projected_attention_output, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Attention residual stream" values={windowed(trace.attention_residual_output, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
      <Heatmap title="Normalized FFN input" values={windowed(trace.normalized_feed_forward_input, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    </div>;
  }

  if (stage === "ffn") {
    const actualColumns = dimensionLabels(dimensionStart, Math.min(dimensionCount, Math.max(0, matrixWidth(trace.gelu_activations) - dimensionStart)), "h");
    const outputDimensionCount = Math.min(dimensionCount, Math.max(0, matrixWidth(trace.feed_forward_output) - dimensionStart));
    return <div className="trace-stack">
      <Heatmap title="FFN pre-activations" values={windowed(trace.feed_forward_pre_activations, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={actualColumns} />
      <Heatmap title="GELU activations" values={windowed(trace.gelu_activations, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={actualColumns} />
      <Heatmap title="FFN output contribution" values={windowed(trace.feed_forward_output, dimensionStart, outputDimensionCount)} rowLabels={rowLabels} columnLabels={dimensionLabels(dimensionStart, outputDimensionCount)} />
      <StatusNotice kind="info">A strongly active hidden neuron is a numerical feature response; it is not automatically a human-interpretable concept.</StatusNotice>
    </div>;
  }

  return <div className="trace-stack">
    <Heatmap title="Block output" values={windowed(trace.block_output, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    <Heatmap title="Final hidden states" values={windowed(response.final_hidden_states, dimensionStart, dimensionCount)} rowLabels={rowLabels} columnLabels={columns} />
    <ProbabilityBars predictions={response.top_predictions} showTechnicalValues />
  </div>;
}

type CausalMaskTableProps = {
  allowed: ReadonlyArray<ReadonlyArray<boolean>>;
  rowLabels: readonly string[];
  columnLabels: readonly string[];
  selected: HeatmapSelection;
  onSelect: (selection: HeatmapSelection) => void;
};

export function CausalMaskTable({ allowed, rowLabels, columnLabels, selected, onSelect }: CausalMaskTableProps) {
  return (
    <div className="heatmap-region causal-mask-region" role="group" aria-label="Causal mask result">
      <div className="matrix-scroll">
        <table className="heatmap-table causal-mask-table">
          <caption>Causal mask result: visible keys and blocked future keys</caption>
          <thead><tr><th scope="col">Query \ Key</th>{columnLabels.map((label, column) => <th scope="col" key={`${label}-${column}`}>{label}</th>)}</tr></thead>
          <tbody>
            {allowed.map((row, rowIndex) => <tr key={rowIndex}>
              <th scope="row">{rowLabels[rowIndex] ?? rowIndex}</th>
              {row.map((isAllowed, columnIndex) => {
                const isSelected = selected.row === rowIndex && selected.column === columnIndex;
                return <td key={columnIndex}>
                  <button
                    type="button"
                    className={`heatmap-cell${isAllowed ? " heatmap-cell--visible" : " heatmap-cell--masked"}${isSelected ? " heatmap-cell--selected" : ""}`}
                    data-masked={isAllowed ? "false" : "true"}
                    aria-pressed={isSelected}
                    aria-label={`${rowLabels[rowIndex] ?? `query ${rowIndex}`} to ${columnLabels[columnIndex] ?? `key ${columnIndex}`}: ${isAllowed ? "Visible" : "Masked future key, probability zero"}`}
                    onClick={() => onSelect({ row: rowIndex, column: columnIndex })}
                  >
                    <span>{isAllowed ? "Visible" : "Masked"}</span>
                    <small>{isAllowed ? "can attend" : "weight 0"}</small>
                  </button>
                </td>;
              })}
            </tr>)}
          </tbody>
        </table>
      </div>
      <p className="visualization-key">Visible = backend mask value true · Masked = backend mask value false · masked probability = 0.</p>
    </div>
  );
}

function AttentionConnections({ response }: { response: TinyInspectResponse }) {
  const connections = useMemo(
    () => response.token_connections.filter((connection) => connection.causally_available),
    [response.token_connections],
  );
  const span = Math.max(1, connections.length - 1);
  return (
    <div className="attention-connections">
      <svg viewBox="0 0 760 235" role="img" aria-labelledby="trained-arcs-title">
        <title id="trained-arcs-title">Attention arcs from {response.tokens[response.selection.query_token]}</title>
        <desc>Arc width and opacity encode the selected head’s attention probability.</desc>
        {connections.map((connection, index) => {
          const x = 70 + index / span * 620;
          return <g key={connection.key_position}>
            <path d={`M 380 52 Q ${(380 + x) / 2} 130 ${x} 178`} fill="none" stroke="var(--color-attention, #74ebb2)" strokeWidth={1 + connection.attention_weight * 13} opacity={0.18 + connection.attention_weight * 0.78} />
            <circle cx={x} cy="192" r="25" />
            <text x={x} y="197" textAnchor="middle">{tokenLabel(connection.key_token)}</text>
          </g>;
        })}
        <circle cx="380" cy="40" r="29" className="attention-query-node" />
        <text x="380" y="45" textAnchor="middle">{tokenLabel(response.tokens[response.selection.query_token] ?? "")}</text>
      </svg>
      <ul className="connection-values" aria-label="Exact attention connections">
        {connections.map((connection) => <li key={connection.key_position}><code>{tokenLabel(connection.key_token)}</code><span>{connection.attention_weight.toFixed(4)}</span></li>)}
      </ul>
    </div>
  );
}
