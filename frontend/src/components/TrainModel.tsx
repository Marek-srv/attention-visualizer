import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { ModelConfig, ModelStatus, TrainingMetric, TrainingStatus } from "../types/api";
import LineChart from "./common/LineChart";
import StatusNotice from "./common/StatusNotice";

const DEFAULT_MODEL: ModelConfig = {
  context_length: 16,
  d_model: 32,
  number_of_heads: 4,
  number_of_layers: 2,
  feed_forward_dimension: 64,
  dropout: 0.1,
};

const IDLE_STATUS: TrainingStatus = { status: "idle", history: [] };

function finite(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

export default function TrainModel() {
  const [epochs, setEpochs] = useState(100);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.003);
  const [status, setStatus] = useState<TrainingStatus>(IDLE_STATUS);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const next = await api.trainingStatus();
    setStatus(next);
    return next;
  }, []);

  const refreshModel = useCallback(async () => {
    const next = await api.modelStatus();
    setModelStatus(next);
  }, []);

  useEffect(() => {
    let active = true;
    Promise.all([api.trainingStatus(), api.modelStatus()])
      .then(([nextStatus, nextModel]) => {
        if (!active) return;
        setStatus(nextStatus);
        setModelStatus(nextModel);
      })
      .catch((requestError: unknown) => active && setError(requestError instanceof Error ? requestError.message : "Unable to read the training service."));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (status.status !== "running") return;
    let active = true;
    let timer: number | undefined;

    const schedule = () => {
      timer = window.setTimeout(async () => {
        try {
          const next = await refresh();
          if (!active) return;
          setError("");
          if (next.status === "running") {
            schedule();
          } else {
            await refreshModel();
          }
        } catch (requestError: unknown) {
          if (!active) return;
          setError(requestError instanceof Error ? requestError.message : "Training status could not be refreshed.");
          schedule();
        }
      }, 1200);
    };

    schedule();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [refresh, refreshModel, status.status]);

  const history: TrainingMetric[] = useMemo(() => status.history ?? status.metrics ?? [], [status.history, status.metrics]);
  const modelConfig = status.model_config ?? modelStatus?.model_config ?? DEFAULT_MODEL;
  const currentEpoch = status.current_epoch ?? status.latest_completed_epoch ?? history.at(-1)?.epoch ?? 0;
  const totalEpochs = status.total_epochs ?? (status.status === "running" ? epochs : Math.max(currentEpoch, epochs));
  const bestValidationLoss = status.best_validation_loss ?? (history.length ? Math.min(...history.map((item) => item.validation_loss)) : null);
  const checkpointAvailable = status.checkpoint_available ?? status.checkpoint_saved ?? modelStatus?.checkpoint_available ?? modelStatus?.checkpoint_exists ?? modelStatus?.available ?? false;

  async function startTraining(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setIsStarting(true);
    try {
      const next = await api.startTraining({ epochs, batch_size: batchSize, learning_rate: learningRate });
      setStatus(next);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Training could not be started.");
    } finally {
      setIsStarting(false);
    }
  }

  async function cancelTraining() {
    setError("");
    setIsCancelling(true);
    try {
      const next = await api.cancelTraining();
      setStatus(next);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Cancellation could not be requested.");
    } finally {
      setIsCancelling(false);
    }
  }

  async function loadCheckpoint() {
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

  const statusKind = status.status === "failed" ? "error" : status.status === "completed" ? "success" : status.status === "cancelled" ? "warning" : "info";

  return (
    <div className="mode-page train-page">
      <header className="mode-hero"><p className="eyebrow">Trainable Language Model Lab</p><h1>Train a tiny Transformer locally.</h1><p>Learn from a deliberately small educational corpus, watch loss change after each epoch, and save the best checkpoint for inspection and prediction.</p></header>

      <section className="lab-panel" aria-labelledby="architecture-title">
        <div className="panel-heading"><div><p className="step-label">Architecture</p><h2 id="architecture-title">Modern pre-normalization decoder</h2></div><span className="phase-badge">Trainable weights</span></div>
        <div className="architecture-callout"><strong>x = x + Attention(LayerNorm(x))</strong><strong>x = x + FFN(LayerNorm(x))</strong><p>The trainable model is separate from the fixed four-dimensional, post-normalization Toy Math Lab.</p></div>
        <dl className="summary-grid">
          <div><dt>Context</dt><dd>{modelConfig.context_length} tokens</dd></div>
          <div><dt>Model width</dt><dd>{modelConfig.d_model}</dd></div>
          <div><dt>Heads</dt><dd>{modelConfig.number_of_heads}</dd></div>
          <div><dt>Layers</dt><dd>{modelConfig.number_of_layers}</dd></div>
          <div><dt>FFN width</dt><dd>{modelConfig.feed_forward_dimension}</dd></div>
          <div><dt>Dropout</dt><dd>{modelConfig.dropout}</dd></div>
        </dl>
      </section>

      <section className="lab-panel" aria-labelledby="training-title">
        <div className="panel-heading"><div><p className="step-label">Local CPU-safe job</p><h2 id="training-title">Training controls</h2></div><span className={`job-pill job-pill--${status.status}`}>{status.status}</span></div>
        <form className="control-form" onSubmit={startTraining}>
          <label>Epochs<input type="number" min="1" max="500" step="1" value={epochs} disabled={status.status === "running"} onChange={(event) => setEpochs(event.currentTarget.valueAsNumber)} /></label>
          <label>Batch size<input type="number" min="1" max="256" step="1" value={batchSize} disabled={status.status === "running"} onChange={(event) => setBatchSize(event.currentTarget.valueAsNumber)} /></label>
          <label>Learning rate<input type="number" min="0.000001" max="1" step="0.0001" value={learningRate} disabled={status.status === "running"} onChange={(event) => setLearningRate(event.currentTarget.valueAsNumber)} /></label>
          <div className="form-actions"><button type="submit" disabled={status.status === "running" || isStarting}>{isStarting ? "Starting…" : "Start training"}</button><button className="secondary-button" type="button" disabled={status.status !== "running" || isCancelling} onClick={() => void cancelTraining()}>{isCancelling ? "Requesting…" : "Cancel"}</button></div>
        </form>
        {error && <StatusNotice kind="error">{error}</StatusNotice>}
        <StatusNotice kind={statusKind}><strong>{status.status === "running" ? `Training epoch ${currentEpoch} of ${totalEpochs}` : `Job status: ${status.status}`}</strong>{status.message && <span>{status.message}</span>}{status.error && <span>{status.error}</span>}</StatusNotice>
        <div className="progress-track" role="progressbar" aria-label="Training epochs" aria-valuemin={0} aria-valuemax={Math.max(totalEpochs, 1)} aria-valuenow={Math.min(currentEpoch, Math.max(totalEpochs, 1))}><span style={{ width: `${Math.min(100, totalEpochs ? currentEpoch / totalEpochs * 100 : 0)}%` }} /></div>
        <dl className="summary-grid training-summary">
          <div><dt>Latest epoch</dt><dd>{currentEpoch} / {totalEpochs}</dd></div>
          <div><dt>Best validation loss</dt><dd>{finite(bestValidationLoss)}</dd></div>
          <div><dt>Checkpoint</dt><dd>{checkpointAvailable ? "Available" : "Not saved yet"}</dd></div>
          <div><dt>Loaded model</dt><dd>{modelStatus?.loaded ? "Ready" : "Not loaded"}</dd></div>
        </dl>
        <div className="inline-actions"><button type="button" className="secondary-button" disabled={!checkpointAvailable || isLoadingModel || status.status === "running"} onClick={() => void loadCheckpoint()}>{isLoadingModel ? "Loading…" : "Load best checkpoint"}</button><button type="button" className="ghost-button" onClick={() => void refresh().catch((requestError: unknown) => setError(requestError instanceof Error ? requestError.message : "Refresh failed."))}>Refresh status</button></div>
      </section>

      <section className="lab-panel chart-section" aria-labelledby="metrics-title">
        <div className="panel-heading"><div><p className="step-label">Completed epochs</p><h2 id="metrics-title">Training history</h2></div><span className="shape-badge">{history.length} points</span></div>
        <div className="chart-grid">
          <article><h3>Cross-entropy loss</h3><LineChart title="Training and validation cross-entropy loss" xLabels={history.map((item) => item.epoch)} series={[{ label: "Training", values: history.map((item) => item.training_loss), color: "#74ebb2" }, { label: "Validation", values: history.map((item) => item.validation_loss), color: "#e8b76a" }]} /></article>
          <article><h3>Perplexity</h3><LineChart title="Training and validation perplexity" xLabels={history.map((item) => item.epoch)} series={[{ label: "Training", values: history.map((item) => item.training_perplexity), color: "#74ebb2" }, { label: "Validation", values: history.map((item) => item.validation_perplexity), color: "#b7a0ff" }]} valueFormatter={(value) => value.toFixed(2)} /></article>
        </div>
      </section>
    </div>
  );
}
