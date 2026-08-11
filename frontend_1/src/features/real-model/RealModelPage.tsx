import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../../api/endpoints";
import { useWorkspace } from "../../app/workspaceContext";
import Heatmap, { type HeatmapSelection } from "../../components/common/Heatmap";
import ProbabilityBars from "../../components/common/ProbabilityBars";
import StatusNotice from "../../components/common/StatusNotice";
import type { PretrainedInspectResponse, PretrainedPredictionResponse, PretrainedStatus } from "../../types/api";

function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function tokenLabel(token: string): string {
  return token.length === 0 ? "<empty>" : token.replaceAll(" ", "␠").replaceAll("\n", "↵");
}

export default function RealModelPage() {
  const workspace = useWorkspace();
  const observedRunNonce = useRef(workspace.runNonce);
  const [status, setStatus] = useState<PretrainedStatus | null>(null);
  const [trace, setTrace] = useState<PretrainedInspectResponse | null>(null);
  const [prediction, setPrediction] = useState<PretrainedPredictionResponse | null>(null);
  const [prompt, setPrompt] = useState(workspace.prompt);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [queryToken, setQueryToken] = useState(0);
  const [selectedCell, setSelectedCell] = useState<HeatmapSelection>({ row: 0, column: 0 });
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(1);
  const [loading, setLoading] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const inspectRequest = useRef<AbortController | null>(null);
  const predictRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.pretrainedStatus({ signal: controller.signal })
      .then(setStatus)
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Pretrained model status is unavailable."));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => () => {
    inspectRequest.current?.abort();
    predictRequest.current?.abort();
  }, []);

  const pretrainedLoadState = status?.status;
  useEffect(() => {
    if (pretrainedLoadState !== "loading") return undefined;
    let active = true;
    let timer: number | undefined;
    let controller: AbortController | undefined;

    const schedule = () => {
      timer = window.setTimeout(async () => {
        controller = new AbortController();
        try {
          const next = await api.pretrainedStatus({ signal: controller.signal });
          if (!active) return;
          setStatus(next);
          if (next.status === "loading") schedule();
        } catch (requestError: unknown) {
          if (!active || controller.signal.aborted) return;
          setError(messageFrom(requestError, "Pretrained model status could not be refreshed."));
          schedule();
        }
      }, 1200);
    };

    schedule();
    return () => {
      active = false;
      controller?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [pretrainedLoadState]);

  const model = status?.model;
  const layerCount = model?.number_of_layers ?? 1;
  const headCount = model?.number_of_heads ?? 1;
  const tokenCount = trace?.tokens.length ?? 1;
  const loaded = status?.status === "loaded" && status.loaded;
  const loadingState = loading || status?.status === "loading";
  const failed = Boolean(loadError) || status?.status === "failed";

  async function loadModel() {
    setLoading(true);
    setLoadError("");
    setError("");
    try {
      setStatus(await api.loadPretrained({}));
    } catch (requestError: unknown) {
      setLoadError(messageFrom(requestError, "The pretrained model could not be loaded."));
    } finally {
      setLoading(false);
    }
  }

  const inspect = useCallback(async (nextQuery = queryToken, nextPrompt = prompt) => {
    inspectRequest.current?.abort();
    const controller = new AbortController();
    inspectRequest.current = controller;
    setInspecting(true);
    setError("");
    try {
      const next = await api.inspectPretrained({ text: nextPrompt, layer, head, query_token: nextQuery, top_k: topK }, { signal: controller.signal });
      setTrace(next);
      setLayer(next.selected_layer);
      setHead(next.selected_head);
      setQueryToken(next.selected_query_index);
      setSelectedCell({ row: next.selected_query_index, column: Math.min(selectedCell.column, next.tokens.length - 1) });
    } catch (requestError: unknown) {
      if (!controller.signal.aborted) setError(messageFrom(requestError, "Pretrained attention inspection failed."));
    } finally {
      if (inspectRequest.current === controller) {
        inspectRequest.current = null;
        setInspecting(false);
      }
    }
  }, [head, layer, prompt, queryToken, selectedCell.column, topK]);

  async function predict() {
    predictRequest.current?.abort();
    const controller = new AbortController();
    predictRequest.current = controller;
    setPredicting(true);
    setError("");
    try {
      setPrediction(await api.predictPretrained({ text: prompt, top_k: topK, temperature }, { signal: controller.signal }));
    } catch (requestError: unknown) {
      if (!controller.signal.aborted) setError(messageFrom(requestError, "Pretrained prediction failed."));
    } finally {
      if (predictRequest.current === controller) {
        predictRequest.current = null;
        setPredicting(false);
      }
    }
  }

  function selectCell(selection: HeatmapSelection) {
    setSelectedCell(selection);
    if (selection.row !== trace?.selected_query_index) {
      setQueryToken(selection.row);
      void inspect(selection.row);
    }
  }

  useEffect(() => {
    if (observedRunNonce.current === workspace.runNonce) return;
    observedRunNonce.current = workspace.runNonce;
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      setPrompt(workspace.prompt);
      if (loaded) void inspect(queryToken, workspace.prompt);
    });
    return () => { active = false; };
  }, [inspect, loaded, queryToken, workspace.prompt, workspace.runNonce]);

  const futureMask = useMemo(() => trace?.attention_matrix.map((row, rowIndex) => row.map((_, columnIndex) => columnIndex > rowIndex)) ?? [], [trace]);
  const selectedProbability = trace?.attention_matrix[selectedCell.row]?.[selectedCell.column];

  return (
    <main className="feature-page real-model-page" aria-labelledby="real-model-title">
      <header className="feature-heading">
        <p className="eyebrow">Optional pretrained weights</p>
        <h1 id="real-model-title">Inspect a small real causal model.</h1>
        <p>The backend reports status on entry, but downloading and loading weights always requires your explicit action.</p>
      </header>

      <section className="workspace-section pretrained-load-state" aria-labelledby="pretrained-status-title" aria-live="polite">
        <div className="section-heading">
          <div><p className="section-kicker">Lazy model service</p><h2 id="pretrained-status-title">{status?.model_name ?? "Configured pretrained model"}</h2></div>
          <span className={`state-badge state-badge--${failed ? "failed" : loadingState ? "loading" : status?.status ?? "not-loaded"}`}>
            {failed ? "failed" : loadingState ? "loading" : status?.status?.replaceAll("_", " ") ?? "checking"}
          </span>
        </div>
        <dl className="model-facts">
          <div><dt>Configured model</dt><dd>{model?.name ?? status?.model_name ?? "Waiting for backend"}</dd></div>
          <div><dt>Device</dt><dd>{model?.device ?? status?.device ?? "CPU by default"}</dd></div>
          <div><dt>Layers</dt><dd>{model?.number_of_layers ?? "—"}</dd></div>
          <div><dt>Heads</dt><dd>{model?.number_of_heads ?? "—"}</dd></div>
        </dl>
        {!loaded || failed ? (
          <button type="button" disabled={loadingState || status === null} onClick={() => void loadModel()}>
            {loadingState ? "Loading model…" : failed ? "Retry loading model" : "Load pretrained model"}
          </button>
        ) : <StatusNotice kind="success">Model loaded. Request only the layer and head you want to inspect.</StatusNotice>}
        {loadError || status?.error ? <StatusNotice kind="error" title="Model loading failed">{loadError || status?.error}</StatusNotice> : null}
      </section>

      {error ? <StatusNotice kind="error">{error}</StatusNotice> : null}

      <section className="workspace-section real-model-controls" aria-labelledby="real-controls-title">
        <div className="section-heading"><div><p className="section-kicker">Selected slice</p><h2 id="real-controls-title">Prompt, layer and head</h2></div></div>
        <label className="prompt-control">Text<textarea rows={2} maxLength={500} value={prompt} onChange={(event) => { setPrompt(event.currentTarget.value); workspace.setPrompt(event.currentTarget.value); }} /></label>
        <div className="control-grid">
          <label>Layer<select value={Math.min(layer, layerCount - 1)} disabled={!loaded} onChange={(event) => setLayer(Number(event.currentTarget.value))}>{Array.from({ length: layerCount }, (_, index) => <option value={index} key={index}>Layer {index}</option>)}</select></label>
          <label>Head<select value={Math.min(head, headCount - 1)} disabled={!loaded} onChange={(event) => setHead(Number(event.currentTarget.value))}>{Array.from({ length: headCount }, (_, index) => <option value={index} key={index}>Head {index}</option>)}</select></label>
          <label>Query token<select value={Math.min(queryToken, tokenCount - 1)} disabled={!trace} onChange={(event) => setQueryToken(Number(event.currentTarget.value))}>{Array.from({ length: tokenCount }, (_, index) => <option value={index} key={index}>{trace ? `${index}: ${tokenLabel(trace.tokens[index] ?? "")}` : index}</option>)}</select></label>
          <label>Top k<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.currentTarget.value))} /></label>
          <button type="button" disabled={!loaded || inspecting || !prompt.trim()} onClick={() => void inspect()}>{inspecting ? "Inspecting…" : "Inspect attention"}</button>
        </div>
      </section>

      <section className="workspace-section real-attention" aria-labelledby="real-attention-title">
        <div className="section-heading"><div><p className="section-kicker">Pretrained trace</p><h2 id="real-attention-title">Selected attention matrix</h2></div>{trace ? <span>Layer {trace.selected_layer} · Head {trace.selected_head}</span> : null}</div>
        {trace ? (
          <>
            <div className="token-sequence" aria-label="Pretrained subword tokens">
              {trace.tokens.map((token, index) => <button type="button" aria-pressed={trace.selected_query_index === index} key={`${trace.token_ids[index]}-${index}`} onClick={() => { setQueryToken(index); void inspect(index); }}><code>{tokenLabel(token)}</code><small>ID {trace.token_ids[index]}</small></button>)}
            </div>
            <Heatmap title="Pretrained attention probabilities" values={trace.attention_matrix} rowLabels={trace.tokens.map(tokenLabel)} columnLabels={trace.tokens.map(tokenLabel)} mask={futureMask} selected={selectedCell} onSelect={selectCell} />
            <p className="row-sum-validation">Query row sum: <strong>{trace.attention_row_sum.toFixed(6)}</strong> · {Math.abs(trace.attention_row_sum - 1) <= 0.0001 ? "approximately 1 ✓" : "check normalization"}</p>
            {selectedProbability !== undefined ? <p className="selected-cell-value">Selected connection: {tokenLabel(trace.tokens[selectedCell.row] ?? "")} → {tokenLabel(trace.tokens[selectedCell.column] ?? "")} = <strong>{selectedProbability.toFixed(6)}</strong></p> : null}
            <PretrainedConnections trace={trace} />
            <ProbabilityBars predictions={trace.top_predictions} showTechnicalValues label="Pretrained next-token candidates from inspection" />
            <StatusNotice kind="warning">Attention is a reading pattern for one head, not a complete measure of token importance or a full explanation of a prediction.</StatusNotice>
          </>
        ) : <p className="empty-state">The real model remains unloaded until requested. After loading, inspect one layer and head at a time.</p>}
      </section>

      <section className="workspace-section real-prediction" aria-labelledby="real-prediction-title">
        <div className="section-heading"><div><p className="section-kicker">Vocabulary projection</p><h2 id="real-prediction-title">Temperature-adjusted prediction</h2></div></div>
        <div className="control-grid">
          <label>Temperature<input type="number" min="0.01" max="10" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.currentTarget.value))} /></label>
          <button type="button" disabled={!loaded || predicting || !prompt.trim()} onClick={() => void predict()}>{predicting ? "Predicting…" : "Predict next token"}</button>
        </div>
        {prediction ? <><ProbabilityBars predictions={prediction.predictions} showTechnicalValues /><p className="interpretation-warning">Percentages are model probabilities, not factual confidence.</p></> : <p className="empty-state">Run prediction after the explicit model load.</p>}
      </section>
    </main>
  );
}

function PretrainedConnections({ trace }: { trace: PretrainedInspectResponse }) {
  const visible = trace.connections.filter((connection) => !connection.is_future);
  const span = Math.max(1, visible.length - 1);
  return (
    <div className="attention-connections">
      <svg viewBox="0 0 760 235" role="img" aria-labelledby="real-arcs-title">
        <title id="real-arcs-title">Pretrained attention arcs from {trace.selected_query_token}</title>
        <desc>Width and opacity encode the actual attention probability for the selected query.</desc>
        {visible.map((connection, index) => {
          const x = 70 + index / span * 620;
          return <g key={connection.key_index}>
            <path d={`M 380 52 Q ${(380 + x) / 2} 130 ${x} 178`} fill="none" stroke="var(--color-attention, #74ebb2)" strokeWidth={1 + connection.attention_weight * 13} opacity={0.18 + connection.attention_weight * 0.78} />
            <circle cx={x} cy="192" r="25" />
            <text x={x} y="197" textAnchor="middle">{tokenLabel(connection.key_token)}</text>
          </g>;
        })}
        <circle cx="380" cy="40" r="29" className="attention-query-node" />
        <text x="380" y="45" textAnchor="middle">{tokenLabel(trace.selected_query_token)}</text>
      </svg>
      <ul className="connection-values" aria-label="Exact pretrained attention connections">
        {visible.map((connection) => <li key={connection.key_index}><code>{tokenLabel(connection.key_token)}</code><span>{connection.attention_weight.toFixed(6)}</span></li>)}
      </ul>
    </div>
  );
}
