type ModelLoadGateProps = {
  loaded: boolean;
  loading: boolean;
  checkpointAvailable?: boolean;
  device?: string;
  onLoad: () => void;
};

export default function ModelLoadGate({ loaded, loading, checkpointAvailable, device, onLoad }: ModelLoadGateProps) {
  return (
    <section className="model-load-gate" aria-live="polite" aria-label="Local trained model status">
      <div>
        <span className={`readiness-indicator${loaded ? " readiness-indicator--ready" : ""}`} aria-hidden="true" />
        <strong>{loaded ? "Local checkpoint loaded" : "Local checkpoint not loaded"}</strong>
        <p>
          {loaded
            ? `Inference uses the saved local model${device ? ` on ${device}` : ""}.`
            : checkpointAvailable === false
              ? "Train the tiny model before loading a checkpoint."
              : "Loading is explicit; this page will not load model weights automatically."}
        </p>
      </div>
      <button type="button" disabled={loading || loaded || checkpointAvailable === false} onClick={onLoad}>
        {loading ? "Loading checkpoint…" : "Load best checkpoint"}
      </button>
    </section>
  );
}
