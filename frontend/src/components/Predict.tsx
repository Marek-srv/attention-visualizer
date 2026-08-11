import { FormEvent, useEffect, useState } from "react";

import { api } from "../api/client";
import type { GenerationResponse, GenerationStrategy, ModelStatus, PredictionResponse } from "../types/api";
import ProbabilityBars from "./common/ProbabilityBars";
import StatusNotice from "./common/StatusNotice";

export default function Predict() {
  const [text, setText] = useState("I love");
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(1);
  const [maxNewTokens, setMaxNewTokens] = useState(8);
  const [strategy, setStrategy] = useState<GenerationStrategy>("greedy");
  const [seed, setSeed] = useState(42);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [generation, setGeneration] = useState<GenerationResponse | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api.modelStatus().then((next) => active && setModelStatus(next)).catch((requestError: unknown) => active && setError(requestError instanceof Error ? requestError.message : "Unable to read model status."));
    return () => { active = false; };
  }, []);

  async function loadModel() {
    setError("");
    setIsLoadingModel(true);
    try {
      setModelStatus(await api.loadModel());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The checkpoint could not be loaded.");
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function predict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsPredicting(true);
    try {
      setPrediction(await api.predict({ text, top_k: topK, temperature }));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction failed.");
    } finally {
      setIsPredicting(false);
    }
  }

  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsGenerating(true);
    try {
      setGeneration(await api.generate({ text, top_k: topK, temperature, max_new_tokens: maxNewTokens, strategy, seed }));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Generation failed.");
    } finally {
      setIsGenerating(false);
    }
  }

  const probabilitySumValid = prediction ? Math.abs(prediction.probability_sum - 1) <= 0.0001 : false;

  return (
    <div className="mode-page predict-page">
      <header className="mode-hero"><p className="eyebrow">Trainable Language Model Lab</p><h1>Predict the next token.</h1><p>Use the locally trained tiny model. Probabilities describe the model’s learned distribution, not factual confidence.</p></header>

      <section className="model-ready-strip" aria-live="polite">
        <div><span className={`readiness-dot ${modelStatus?.loaded ? "ready" : ""}`} aria-hidden="true" /><strong>{modelStatus?.loaded ? "Tiny model loaded" : "No tiny model loaded"}</strong><small>{modelStatus?.vocabulary_size ? `${modelStatus.vocabulary_size}-token vocabulary` : "Train and load a checkpoint before inference."}</small></div>
        <button type="button" className="secondary-button" disabled={isLoadingModel || modelStatus?.loaded} onClick={() => void loadModel()}>{isLoadingModel ? "Loading…" : "Load best checkpoint"}</button>
      </section>

      {error && <StatusNotice kind="error">{error}</StatusNotice>}

      <section className="lab-panel" aria-labelledby="prediction-title">
        <div className="panel-heading"><div><p className="step-label">One forward pass</p><h2 id="prediction-title">Top next-token probabilities</h2></div><span className="phase-badge">Trained weights</span></div>
        <form className="predict-form" onSubmit={predict}>
          <label className="prompt-field">Prompt<textarea value={text} maxLength={500} rows={3} onChange={(event) => setText(event.currentTarget.value)} placeholder="Try: I love" /><small>{text.length} / 500 characters</small></label>
          <div className="compact-controls"><label>Top k<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(event.currentTarget.valueAsNumber)} /></label><label>Temperature<input type="number" min="0.01" max="10" step="0.05" value={temperature} onChange={(event) => setTemperature(event.currentTarget.valueAsNumber)} /></label><button type="submit" disabled={isPredicting || !text.trim()}>{isPredicting ? "Predicting…" : "Predict next token"}</button></div>
        </form>
        {prediction ? <div className="prediction-results" aria-live="polite"><div className="tokenized-prompt"><span>Model input</span>{prediction.tokens.map((token, index) => <b key={`${token}-${index}`}>{token}<small>ID {prediction.token_ids[index]}</small></b>)}</div><ProbabilityBars predictions={prediction.predictions} /><div className={`sum-check ${probabilitySumValid ? "valid" : "invalid"}`}><strong>{probabilitySumValid ? "✓" : "!"} Full vocabulary probability sum</strong><span>{prediction.probability_sum.toFixed(6)}</span></div></div> : <div className="empty-state">Enter a prompt and run prediction. The bars will show temperature-adjusted model probabilities.</div>}
      </section>

      <section className="lab-panel" aria-labelledby="generation-title">
        <div className="panel-heading"><div><p className="step-label">Autoregressive loop</p><h2 id="generation-title">Generate a short continuation</h2></div><span className="shape-badge">Maximum 50 new tokens</span></div>
        <form className="generation-controls" onSubmit={generate}>
          <label>Maximum new tokens<input type="number" min="1" max="50" value={maxNewTokens} onChange={(event) => setMaxNewTokens(event.currentTarget.valueAsNumber)} /></label>
          <label>Strategy<select value={strategy} onChange={(event) => setStrategy(event.currentTarget.value as GenerationStrategy)}><option value="greedy">Greedy</option><option value="sample">Sample</option></select></label>
          <label>Sampling seed<input type="number" min="0" max="2147483647" value={seed} disabled={strategy === "greedy"} onChange={(event) => setSeed(event.currentTarget.valueAsNumber)} /></label>
          <button type="submit" disabled={isGenerating || !text.trim()}>{isGenerating ? "Generating…" : "Generate"}</button>
        </form>
        {generation ? <div className="generation-results" aria-live="polite"><div className="generated-copy"><span>Generated text</span><p>{generation.generated_text}</p>{generation.stop_reason && <small>Stopped: {generation.stop_reason}</small>}</div><ol className="generation-steps">{generation.steps.map((step, index) => <li key={`${step.step}-${step.chosen_token_id}-${index}`}><div className="chosen-token"><span>Step {step.step}</span><strong>{step.chosen_token}</strong><small>ID {step.chosen_token_id}{typeof step.chosen_probability === "number" ? ` · ${(step.chosen_probability * 100).toFixed(2)}%` : ""}</small></div><ProbabilityBars predictions={step.top_predictions} emptyLabel="No candidate distribution returned." /></li>)}</ol></div> : <div className="empty-state compact">Generation stops at &lt;EOS&gt; or the requested limit. &lt;PAD&gt; and &lt;BOS&gt; are never shown as generated text.</div>}
      </section>
    </div>
  );
}
