import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../../api/endpoints";
import { useWorkspace } from "../../app/workspaceContext";
import ModelLoadGate from "../../components/common/ModelLoadGate";
import ProbabilityBars, { type ProbabilityDatum } from "../../components/common/ProbabilityBars";
import StatusNotice from "../../components/common/StatusNotice";
import type { GenerationResponse, GenerationStrategy, ModelStatus, PredictionResponse } from "../../types/api";

function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function tokenLabel(token: string): string {
  return token.length === 0 ? "<empty>" : token.replaceAll(" ", "␠").replaceAll("\n", "↵");
}

function initiallyReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function PredictionPage() {
  const workspace = useWorkspace();
  const observedRunNonce = useRef(workspace.runNonce);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [prompt, setPrompt] = useState(workspace.prompt);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState(5);
  const [maxNewTokens, setMaxNewTokens] = useState(8);
  const [strategy, setStrategy] = useState<GenerationStrategy>("greedy");
  const [seed, setSeed] = useState(42);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [generation, setGeneration] = useState<GenerationResponse | null>(null);
  const [selectedStep, setSelectedStep] = useState(0);
  const [visibleStepCount, setVisibleStepCount] = useState(0);
  const [loadingModel, setLoadingModel] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [reducedMotion, setReducedMotion] = useState(initiallyReducedMotion);
  const predictionRequest = useRef<AbortController | null>(null);
  const generationRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.modelStatus({ signal: controller.signal })
      .then(setModelStatus)
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Unable to read the local model status."));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => () => {
    predictionRequest.current?.abort();
    generationRequest.current?.abort();
  }, []);

  useEffect(() => {
    if (!generation) return undefined;
    if (reducedMotion) {
      const revealTimer = window.setTimeout(() => setVisibleStepCount(generation.steps.length), 0);
      return () => window.clearTimeout(revealTimer);
    }
    if (generation.steps.length <= 1) return undefined;
    const timer = window.setInterval(() => {
      setVisibleStepCount((current) => {
        const next = Math.min(current + 1, generation.steps.length);
        if (next >= generation.steps.length) window.clearInterval(timer);
        return next;
      });
    }, 360);
    return () => window.clearInterval(timer);
  }, [generation, reducedMotion]);

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

  const runPrediction = useCallback(async (nextPrompt = prompt) => {
    predictionRequest.current?.abort();
    const controller = new AbortController();
    predictionRequest.current = controller;
    setPredicting(true);
    setError("");
    try {
      setPrediction(await api.predict({ text: nextPrompt, top_k: topK, temperature }, { signal: controller.signal }));
    } catch (requestError: unknown) {
      if (!controller.signal.aborted) setError(messageFrom(requestError, "Prediction failed."));
    } finally {
      if (predictionRequest.current === controller) {
        predictionRequest.current = null;
        setPredicting(false);
      }
    }
  }, [prompt, temperature, topK]);

  async function runGeneration() {
    generationRequest.current?.abort();
    const controller = new AbortController();
    generationRequest.current = controller;
    setGenerating(true);
    setError("");
    try {
      const next = await api.generate({
        text: prompt,
        top_k: topK,
        temperature,
        max_new_tokens: maxNewTokens,
        strategy,
        seed,
      }, { signal: controller.signal });
      setGeneration(next);
      setSelectedStep(0);
      setVisibleStepCount(reducedMotion ? next.steps.length : Math.min(1, next.steps.length));
    } catch (requestError: unknown) {
      if (!controller.signal.aborted) setError(messageFrom(requestError, "Generation failed."));
    } finally {
      if (generationRequest.current === controller) {
        generationRequest.current = null;
        setGenerating(false);
      }
    }
  }

  useEffect(() => {
    if (observedRunNonce.current === workspace.runNonce) return;
    observedRunNonce.current = workspace.runNonce;
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      setPrompt(workspace.prompt);
      if (modelStatus?.loaded) void runPrediction(workspace.prompt);
    });
    return () => { active = false; };
  }, [modelStatus?.loaded, runPrediction, workspace.prompt, workspace.runNonce]);

  const selectedGenerationStep = generation?.steps[selectedStep];

  return (
    <main className="feature-page prediction-page" aria-labelledby="prediction-title">
      <header className="feature-heading">
        <p className="eyebrow">Local trained weights</p>
        <h1 id="prediction-title">Continue a sequence.</h1>
        <p>Inspect the tiny model’s next-token distribution, then follow an actual generated sequence step by step.</p>
      </header>

      <ModelLoadGate
        loaded={modelStatus?.loaded ?? false}
        loading={loadingModel}
        checkpointAvailable={modelStatus?.checkpoint_available}
        device={modelStatus?.device}
        onLoad={() => void loadCheckpoint()}
      />
      {error ? <StatusNotice kind="error">{error}</StatusNotice> : null}

      <section className="workspace-section prediction-controls" aria-labelledby="prompt-title">
        <div className="section-heading"><div><p className="section-kicker">Prompt</p><h2 id="prompt-title">Inference controls</h2></div></div>
        <label className="prompt-control">
          Text
          <textarea rows={3} maxLength={500} value={prompt} onChange={(event) => { setPrompt(event.currentTarget.value); workspace.setPrompt(event.currentTarget.value); }} />
          <small>{prompt.length} / 500 characters</small>
        </label>
        <div className="control-grid">
          <label>Temperature<input type="number" min="0.01" max="10" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.currentTarget.value))} /></label>
          <label>Top k<input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.currentTarget.value))} /></label>
        </div>
      </section>

      <section className="workspace-section" aria-labelledby="next-token-title">
        <div className="section-heading">
          <div><p className="section-kicker">One forward pass</p><h2 id="next-token-title">Next-token probabilities</h2></div>
          <button type="button" disabled={!modelStatus?.loaded || predicting || !prompt.trim()} onClick={() => void runPrediction()}>{predicting ? "Predicting…" : "Predict next token"}</button>
        </div>
        {prediction ? (
          <>
            <div className="token-sequence" aria-label="Tokenized prediction input">
              {prediction.tokens.map((token, index) => <span key={`${prediction.token_ids[index]}-${index}`}><code>{tokenLabel(token)}</code><small>ID {prediction.token_ids[index]}</small></span>)}
            </div>
            <ProbabilityBars predictions={prediction.predictions} />
            <details>
              <summary>Show token IDs and logits</summary>
              <ProbabilityBars predictions={prediction.predictions} showTechnicalValues label="Next-token candidates with token IDs and logits" />
            </details>
            <p className="interpretation-warning">These percentages are model probabilities, not factual confidence.</p>
          </>
        ) : <p className="empty-state">Load the checkpoint and run a prediction to see candidates.</p>}
      </section>

      <section className="workspace-section" aria-labelledby="generation-title">
        <div className="section-heading"><div><p className="section-kicker">Autoregressive loop</p><h2 id="generation-title">Generate token by token</h2></div></div>
        <div className="control-grid generation-controls">
          <label>Strategy<select value={strategy} onChange={(event) => setStrategy(event.currentTarget.value as GenerationStrategy)}><option value="greedy">Greedy</option><option value="sample">Sample</option></select></label>
          <label>New tokens<input type="number" min="1" max="50" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.currentTarget.value))} /></label>
          <label>Random seed<input type="number" value={seed} disabled={strategy === "greedy"} onChange={(event) => setSeed(Number(event.currentTarget.value))} /></label>
          <button type="button" disabled={!modelStatus?.loaded || generating || !prompt.trim()} onClick={() => void runGeneration()}>{generating ? "Generating…" : "Generate"}</button>
        </div>

        {generation ? (
          <>
            <div className="generation-result" aria-live="polite">
              <span>Generated text</span><strong>{generation.generated_text}</strong><small>Stopped: {generation.stop_reason.replaceAll("_", " ")}</small>
            </div>
            <ol className="generation-timeline" aria-label="Generated-token timeline">
              {generation.steps.slice(0, visibleStepCount).map((step, index) => (
                <li key={step.step}>
                  <button type="button" aria-pressed={selectedStep === index} onClick={() => setSelectedStep(index)}>
                    <span>Step {step.step}</span>
                    <code>{tokenLabel(step.chosen_token)}</code>
                    <small>{(step.chosen_probability * 100).toFixed(2)}%{step.is_eos ? " · EOS" : ""}</small>
                  </button>
                </li>
              ))}
            </ol>
            {selectedGenerationStep ? (
              <div className="generation-step-inspector">
                <h3>Why step {selectedGenerationStep.step} produced <code>{tokenLabel(selectedGenerationStep.chosen_token)}</code></h3>
                <ProbabilityBars
                  predictions={selectedGenerationStep.top_predictions as readonly ProbabilityDatum[]}
                  selectedTokenId={selectedGenerationStep.chosen_token_id}
                  showTechnicalValues
                  label={`Candidates at generation step ${selectedGenerationStep.step}`}
                />
              </div>
            ) : null}
          </>
        ) : <p className="empty-state">Generation steps will appear here using the exact backend response.</p>}
      </section>
    </main>
  );
}
