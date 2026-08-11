import { FormEvent, useEffect, useState } from "react";

import { api } from "../api/client";
import type { PredictionResponse, PretrainedInspectResponse, PretrainedStatus } from "../types/api";
import Heatmap from "./common/Heatmap";
import ProbabilityBars from "./common/ProbabilityBars";
import StatusNotice from "./common/StatusNotice";

function PretrainedGraph({ trace }: { trace: PretrainedInspectResponse }) {
  const span = Math.max(trace.connections.length - 1, 1);
  return <svg className="trained-connection-graph" viewBox="0 0 800 260" role="img" aria-label={`Attention pattern from ${trace.selected_query_token}`}><title>Pretrained head attention pattern. Thickness and opacity represent attention weight.</title>{trace.connections.map((connection, index) => { const x = 65 + index * 670 / span; return <g key={`${connection.key_index}-${connection.key_token_id}`}><line x1="400" y1="65" x2={x} y2="190" stroke={connection.is_future ? "#67746f" : "#74ebb2"} strokeWidth={1 + connection.attention_weight * 14} opacity={connection.is_future ? 0.15 : 0.12 + connection.attention_weight * 0.82} /><text x={(400 + x) / 2} y="125" fill="#d9e6e1" fontSize="13" textAnchor="middle">{connection.attention_weight.toFixed(4)}</text><circle cx={x} cy="205" r="27" fill="#10221e" stroke={connection.is_future ? "#67746f" : "#4da97d"} /><text x={x} y="210" fill="#eff8f4" fontSize="12" textAnchor="middle">{connection.key_token}</text></g>; })}<circle cx="400" cy="48" r="31" fill="#173c30" stroke="#74ebb2" strokeWidth="2" /><text x="400" y="53" fill="#f4faf7" fontSize="13" textAnchor="middle">{trace.selected_query_token}</text></svg>;
}

export default function RealModel() {
  const [status, setStatus] = useState<PretrainedStatus | null>(null);
  const [text, setText] = useState("I love");
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [queryIndex, setQueryIndex] = useState(0);
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(1);
  const [trace, setTrace] = useState<PretrainedInspectResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api.pretrainedStatus().then((next) => active && setStatus(next)).catch((requestError: unknown) => active && setError(requestError instanceof Error ? requestError.message : "Unable to read pretrained-model status."));
    return () => { active = false; };
  }, []);

  async function loadModel() {
    setError("");
    setIsLoadingModel(true);
    try {
      setStatus(await api.loadPretrained());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The pretrained model could not be loaded.");
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function inspect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsInspecting(true);
    try {
      const next = await api.inspectPretrained({ text, layer, head, query_token: queryIndex, top_k: topK });
      setTrace(next);
      setQueryIndex(next.selected_query_index);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Pretrained inspection failed.");
    } finally {
      setIsInspecting(false);
    }
  }

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsPredicting(true);
    try {
      setPrediction(await api.predictPretrained({ text, top_k: topK, temperature }));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Pretrained prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }

  const model = status?.model;
  const layerCount = model?.number_of_layers ?? 2;
  const headCount = model?.number_of_heads ?? 2;

  return (
    <div className="mode-page real-model-page">
      <header className="mode-hero"><p className="eyebrow">Optional pretrained model</p><h1>Inspect a real, small causal model.</h1><p>This mode is separate from both the fixed Toy Math Lab and your locally trained tiny model. It loads lazily and never invents values when a download fails.</p></header>

      <section className="lab-panel real-load-panel" aria-labelledby="real-load-title">
        <div className="panel-heading"><div><p className="step-label">Hugging Face Transformers</p><h2 id="real-load-title">Pretrained model status</h2></div><span className={`job-pill job-pill--${status?.status ?? "idle"}`}>{status?.status?.replace("_", " ") ?? "checking"}</span></div>
        <div className="real-model-summary"><div><span>Configured model</span><strong>{model?.name ?? status?.model_name ?? "sshleifer/tiny-gpt2"}</strong></div><div><span>Device</span><strong>{model?.device ?? status?.device ?? "CPU by default"}</strong></div><div><span>Dependencies</span><strong>{status?.dependencies_available ? status.dependencies_available.transformers && status.dependencies_available.torch ? "Available" : "Missing" : "Checking"}</strong></div><div><span>Attention implementation</span><strong>{model?.attention_implementation ?? "Loaded on demand"}</strong></div></div>
        <p className="download-note">Loading may download model files the first time and can take several minutes. The backend uses <code>sshleifer/tiny-gpt2</code> by default; an environment variable can select <code>distilgpt2</code>.</p>
        <button type="button" disabled={isLoadingModel || status?.loaded} onClick={() => void loadModel()}>{isLoadingModel ? "Loading model…" : status?.loaded ? "Pretrained model loaded" : "Load pretrained model"}</button>
        {error && <StatusNotice kind="error">{error}</StatusNotice>}
        {status?.error && <StatusNotice kind="error">{status.error}</StatusNotice>}
      </section>

      <section className="lab-panel" aria-labelledby="real-inspect-title">
        <div className="panel-heading"><div><p className="step-label">Focused tensor slice</p><h2 id="real-inspect-title">Pretrained attention inspector</h2></div>{model && <span className="shape-badge">{model.number_of_layers} layers · {model.number_of_heads} heads</span>}</div>
        <form className="real-inspect-form" onSubmit={inspect}>
          <label className="prompt-field">Prompt<textarea rows={3} maxLength={500} value={text} onChange={(event) => setText(event.currentTarget.value)} /></label>
          <div className="selector-grid"><label>Layer<select value={layer} onChange={(event) => setLayer(Number(event.currentTarget.value))}>{Array.from({ length: layerCount }, (_, index) => <option value={index} key={index}>Layer {index}</option>)}</select></label><label>Head<select value={head} onChange={(event) => setHead(Number(event.currentTarget.value))}>{Array.from({ length: headCount }, (_, index) => <option value={index} key={index}>Head {index}</option>)}</select></label><label>Query token<select value={queryIndex} onChange={(event) => setQueryIndex(Number(event.currentTarget.value))}>{trace?.tokens.length ? trace.tokens.map((token, index) => <option value={index} key={`${token}-${index}`}>{index}: {token}</option>) : <option value="0">0 — inspect first</option>}</select></label><label>Top k<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(event.currentTarget.valueAsNumber)} /></label></div>
          <button type="submit" disabled={!status?.loaded || isInspecting || !text.trim()}>{isInspecting ? "Inspecting…" : "Inspect pretrained attention"}</button>
        </form>
        {trace ? <div className="pretrained-trace" aria-live="polite"><div className="tokenized-prompt"><span>Subword tokens</span>{trace.tokens.map((token, index) => <b className={index === trace.selected_query_index ? "selected" : ""} key={`${token}-${index}`}>{token}<small>ID {trace.token_ids[index]}</small></b>)}</div>{trace.context_truncated && <StatusNotice kind="warning">The prompt was safely truncated from {trace.original_token_count} to {trace.token_count} tokens.</StatusNotice>}<Heatmap title={`Layer ${trace.selected_layer}, head ${trace.selected_head} attention`} values={trace.attention_matrix} rowLabels={trace.tokens} columnLabels={trace.tokens} selectedRow={trace.selected_query_index} /><div className="selected-readout"><span>Selected query row sum</span><strong>{trace.selected_query_token}</strong><b>{trace.attention_row_sum.toFixed(6)}</b></div><PretrainedGraph trace={trace} /><StatusNotice>{trace.attention_note}</StatusNotice><div className="top-output-probabilities"><h3>Top next-token model probabilities</h3><ProbabilityBars predictions={trace.top_predictions} /><div className={`sum-check ${Math.abs(trace.probability_sum - 1) <= 0.0001 ? "valid" : "invalid"}`}><strong>Full vocabulary probability sum</strong><span>{trace.probability_sum.toFixed(6)}</span></div></div></div> : <div className="empty-state compact">Load the model, then inspect a selected layer, head, and query token.</div>}
      </section>

      <section className="lab-panel" aria-labelledby="real-predict-title">
        <div className="panel-heading"><div><p className="step-label">Vocabulary projection</p><h2 id="real-predict-title">Pretrained next-token prediction</h2></div><span className="phase-badge">Pretrained weights</span></div>
        <form className="compact-controls" onSubmit={predict}><label>Temperature<input type="number" min="0.01" max="10" step="0.05" value={temperature} onChange={(event) => setTemperature(event.currentTarget.valueAsNumber)} /></label><button type="submit" disabled={!status?.loaded || isPredicting || !text.trim()}>{isPredicting ? "Predicting…" : "Predict with pretrained model"}</button></form>
        {prediction ? <div className="prediction-results"><ProbabilityBars predictions={prediction.predictions} /><div className={`sum-check ${Math.abs(prediction.probability_sum - 1) <= 0.0001 ? "valid" : "invalid"}`}><strong>Full vocabulary probability sum</strong><span>{prediction.probability_sum.toFixed(6)}</span></div></div> : <div className="empty-state compact">Predictions here use pretrained model weights, not the local tiny model.</div>}
      </section>
    </div>
  );
}
