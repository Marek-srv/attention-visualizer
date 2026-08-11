import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../../api/endpoints";
import { useWorkspace } from "../../app/workspaceContext";
import LineChart, { type ChartPoint } from "../../components/common/LineChart";
import StatusNotice from "../../components/common/StatusNotice";
import { useTrainingPolling } from "../../hooks/useTrainingPolling";
import type { ModelStatus, TrainingMetric, TrainingStartRequest, TrainingStatus } from "../../types/api";

function messageFrom(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function displayed(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export default function TrainingPage() {
  const { runNonce } = useWorkspace();
  const observedRunNonce = useRef(runNonce);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.003);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [gradientClip, setGradientClip] = useState(1);
  const [seed, setSeed] = useState(42);
  const [contextLength, setContextLength] = useState(16);
  const [modelDimension, setModelDimension] = useState(32);
  const [heads, setHeads] = useState(4);
  const [layers, setLayers] = useState(2);
  const [feedForwardDimension, setFeedForwardDimension] = useState(64);
  const [dropout, setDropout] = useState(0.1);
  const [selectedEpoch, setSelectedEpoch] = useState<number | undefined>();
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [loadingCheckpoint, setLoadingCheckpoint] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      api.trainingStatus({ signal: controller.signal }),
      api.modelStatus({ signal: controller.signal }),
    ])
      .then(([nextStatus, nextModelStatus]) => {
        setStatus(nextStatus);
        setModelStatus(nextModelStatus);
        setSelectedEpoch(nextStatus.history.at(-1)?.epoch);
      })
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Unable to read the training service."));
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (observedRunNonce.current === runNonce) return undefined;
    observedRunNonce.current = runNonce;
    const controller = new AbortController();
    Promise.all([
      api.trainingStatus({ signal: controller.signal }),
      api.modelStatus({ signal: controller.signal }),
    ])
      .then(([nextStatus, nextModelStatus]) => {
        setStatus(nextStatus);
        setModelStatus(nextModelStatus);
        setError("");
      })
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Training status could not be refreshed."));
      });
    return () => controller.abort();
  }, [runNonce]);

  const handlePollingError = useCallback((pollError: unknown) => {
    setError(messageFrom(pollError, "Training progress could not be refreshed."));
  }, []);

  const handlePollingStatus = useCallback((nextStatus: TrainingStatus) => {
    setStatus(nextStatus);
    setSelectedEpoch((current) => current ?? nextStatus.history.at(-1)?.epoch);
    setError("");
  }, []);

  const fetchPollingStatus = useCallback(
    (signal: AbortSignal) => api.trainingStatus({ signal }),
    [],
  );

  useTrainingPolling({
    enabled: status?.status === "running",
    fetchStatus: fetchPollingStatus,
    onStatus: handlePollingStatus,
    onError: handlePollingError,
  });

  const terminalStatus = status?.status;
  useEffect(() => {
    if (!terminalStatus || !["completed", "cancelled", "failed"].includes(terminalStatus)) return undefined;
    const controller = new AbortController();
    api.modelStatus({ signal: controller.signal })
      .then(setModelStatus)
      .catch((requestError: unknown) => {
        if (!controller.signal.aborted) setError(messageFrom(requestError, "Checkpoint status could not be refreshed."));
      });
    return () => controller.abort();
  }, [terminalStatus]);

  const history = useMemo(() => status?.history ?? [], [status?.history]);
  const lossPoints = useMemo<ChartPoint[]>(() => history.map((metric) => ({
    epoch: metric.epoch,
    training: metric.training_loss,
    validation: metric.validation_loss,
  })), [history]);
  const perplexityPoints = useMemo<ChartPoint[]>(() => history.map((metric) => ({
    epoch: metric.epoch,
    training: metric.training_perplexity,
    validation: metric.validation_perplexity,
  })), [history]);
  const selectedMetric = history.find((metric) => metric.epoch === selectedEpoch);
  const running = status?.status === "running";
  const completedEpoch = status?.latest_completed_epoch ?? 0;
  const totalEpochs = status?.total_epochs ?? epochs;
  const progress = totalEpochs > 0 ? Math.min(100, completedEpoch / totalEpochs * 100) : 0;
  const effectiveModel = status?.model_config ?? modelStatus?.model_config;

  async function startTraining() {
    const request: TrainingStartRequest = {
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      weight_decay: weightDecay,
      gradient_clip: gradientClip,
      seed,
      validation_fraction: 0.2,
      model_config: {
        context_length: contextLength,
        d_model: modelDimension,
        number_of_heads: heads,
        number_of_layers: layers,
        feed_forward_dimension: feedForwardDimension,
        dropout,
      },
    };
    setStarting(true);
    setError("");
    try {
      setStatus(await api.startTraining(request));
      setSelectedEpoch(undefined);
    } catch (requestError: unknown) {
      setError(messageFrom(requestError, "Training could not be started."));
    } finally {
      setStarting(false);
    }
  }

  async function cancelTraining() {
    setCancelling(true);
    setError("");
    try {
      setStatus(await api.cancelTraining());
    } catch (requestError: unknown) {
      setError(messageFrom(requestError, "Training cancellation failed."));
    } finally {
      setCancelling(false);
    }
  }

  async function loadCheckpoint() {
    setLoadingCheckpoint(true);
    setError("");
    try {
      setModelStatus(await api.loadModel());
    } catch (requestError: unknown) {
      setError(messageFrom(requestError, "The saved checkpoint could not be loaded."));
    } finally {
      setLoadingCheckpoint(false);
    }
  }

  return (
    <main className="feature-page training-page" aria-labelledby="training-title">
      <header className="feature-heading">
        <p className="eyebrow">Local tiny Transformer</p>
        <h1 id="training-title">Train a decoder, one epoch at a time.</h1>
        <p>Watch optimization metrics update without confusing low training loss with useful language quality.</p>
      </header>

      {error ? <StatusNotice kind="error" title="Training service error">{error}</StatusNotice> : null}

      <section className="workspace-section training-controls" aria-labelledby="configuration-title">
        <div className="section-heading">
          <div><p className="section-kicker">Configuration</p><h2 id="configuration-title">Compact training setup</h2></div>
          <span className={`state-badge state-badge--${status?.status ?? "idle"}`} aria-live="polite">
            {status?.status ?? "idle"}
          </span>
        </div>
        <div className="control-grid">
          <label>Epochs<input type="number" min="1" max="500" value={epochs} disabled={running} onChange={(event) => setEpochs(Number(event.currentTarget.value))} /></label>
          <label>Batch size<input type="number" min="1" max="256" value={batchSize} disabled={running} onChange={(event) => setBatchSize(Number(event.currentTarget.value))} /></label>
          <label>Learning rate<input type="number" min="0.000001" max="1" step="0.0001" value={learningRate} disabled={running} onChange={(event) => setLearningRate(Number(event.currentTarget.value))} /></label>
          <label>Weight decay<input type="number" min="0" max="10" step="0.001" value={weightDecay} disabled={running} onChange={(event) => setWeightDecay(Number(event.currentTarget.value))} /></label>
          <label>Gradient clip<input type="number" min="0.01" max="100" step="0.1" value={gradientClip} disabled={running} onChange={(event) => setGradientClip(Number(event.currentTarget.value))} /></label>
          <label>Seed<input type="number" value={seed} disabled={running} onChange={(event) => setSeed(Number(event.currentTarget.value))} /></label>
        </div>
        <details className="advanced-controls">
          <summary>Model architecture</summary>
          <div className="control-grid">
            <label>Context length<input type="number" min="2" max="512" value={contextLength} disabled={running} onChange={(event) => setContextLength(Number(event.currentTarget.value))} /></label>
            <label>Model dimensions<input type="number" min="4" max="1024" step="4" value={modelDimension} disabled={running} onChange={(event) => setModelDimension(Number(event.currentTarget.value))} /></label>
            <label>Attention heads<input type="number" min="1" max="64" value={heads} disabled={running} onChange={(event) => setHeads(Number(event.currentTarget.value))} /></label>
            <label>Decoder layers<input type="number" min="1" max="24" value={layers} disabled={running} onChange={(event) => setLayers(Number(event.currentTarget.value))} /></label>
            <label>FFN dimensions<input type="number" min="4" max="8192" value={feedForwardDimension} disabled={running} onChange={(event) => setFeedForwardDimension(Number(event.currentTarget.value))} /></label>
            <label>Dropout<input type="number" min="0" max="0.99" step="0.05" value={dropout} disabled={running} onChange={(event) => setDropout(Number(event.currentTarget.value))} /></label>
          </div>
        </details>
        <div className="button-row">
          <button type="button" disabled={running || starting} onClick={() => void startTraining()}>{starting ? "Starting…" : "Start training"}</button>
          <button type="button" className="secondary-button" disabled={!running || cancelling} onClick={() => void cancelTraining()}>{cancelling ? "Cancelling…" : "Cancel"}</button>
        </div>
      </section>

      <section className="workspace-section training-progress" aria-labelledby="progress-title">
        <div className="section-heading">
          <div><p className="section-kicker">Background job</p><h2 id="progress-title">Progress and checkpoint</h2></div>
          <span>{completedEpoch} / {totalEpochs} epochs</span>
        </div>
        <div className="progress-track" role="progressbar" aria-label="Training epoch progress" aria-valuemin={0} aria-valuemax={totalEpochs} aria-valuenow={completedEpoch}>
          <span style={{ width: `${progress}%` }} />
        </div>
        <dl className="metric-summary">
          <div><dt>Best validation loss</dt><dd>{displayed(status?.best_validation_loss)}</dd></div>
          <div><dt>Checkpoint</dt><dd>{status?.checkpoint_available || modelStatus?.checkpoint_available ? "Available" : "Not saved yet"}</dd></div>
          <div><dt>Device</dt><dd>{modelStatus?.device ?? "CPU by default"}</dd></div>
          <div><dt>Architecture</dt><dd>{effectiveModel ? `${effectiveModel.number_of_layers} layers · ${effectiveModel.number_of_heads} heads · d=${effectiveModel.d_model}` : "Not started"}</dd></div>
        </dl>
        <button type="button" className="secondary-button" disabled={loadingCheckpoint || modelStatus?.loaded || !(status?.checkpoint_available || modelStatus?.checkpoint_available)} onClick={() => void loadCheckpoint()}>
          {loadingCheckpoint ? "Loading…" : modelStatus?.loaded ? "Best checkpoint loaded" : "Load best checkpoint"}
        </button>
      </section>

      <section className="workspace-section chart-section" aria-labelledby="metrics-title">
        <div className="section-heading"><div><p className="section-kicker">Metrics</p><h2 id="metrics-title">Loss and perplexity</h2></div></div>
        <div className="chart-grid">
          <article><h3>Loss</h3><LineChart points={lossPoints} title="Loss" valueLabel="Loss" selectedEpoch={selectedEpoch} onSelectEpoch={setSelectedEpoch} /></article>
          <article><h3>Perplexity</h3><LineChart points={perplexityPoints} title="Perplexity" valueLabel="Perplexity" selectedEpoch={selectedEpoch} onSelectEpoch={setSelectedEpoch} /></article>
        </div>
        {selectedMetric ? <EpochDetails metric={selectedMetric} /> : null}
        <StatusNotice kind="warning">Lower loss means the model fits this tiny corpus better; it does not guarantee fluent, factual, or generally useful language.</StatusNotice>
      </section>
    </main>
  );
}

function EpochDetails({ metric }: { metric: TrainingMetric }) {
  return (
    <dl className="selected-epoch-details" aria-label={`Exact metrics for epoch ${metric.epoch}`}>
      <div><dt>Selected epoch</dt><dd>{metric.epoch}</dd></div>
      <div><dt>Training loss</dt><dd>{displayed(metric.training_loss)}</dd></div>
      <div><dt>Validation loss</dt><dd>{displayed(metric.validation_loss)}</dd></div>
      <div><dt>Training perplexity</dt><dd>{displayed(metric.training_perplexity)}</dd></div>
      <div><dt>Validation perplexity</dt><dd>{displayed(metric.validation_perplexity)}</dd></div>
      <div><dt>Learning rate</dt><dd>{displayed(metric.learning_rate, 6)}</dd></div>
    </dl>
  );
}
