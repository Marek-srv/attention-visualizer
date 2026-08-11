import type {
  HealthResponse,
  ModelStatus,
  PredictionResponse,
  PretrainedStatus,
  TrainingStatus,
} from "../types/api";

export function jsonResponse(payload: unknown, status = 200, statusText = "OK"): Response {
  return new Response(JSON.stringify(payload), {
    status,
    statusText,
    headers: { "Content-Type": "application/json" },
  });
}

export function createHealthResponse(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return { status: "ok", service: "attention-visualizer-api", ...overrides };
}

export function createTrainingStatus(overrides: Partial<TrainingStatus> = {}): TrainingStatus {
  return {
    state: "idle",
    status: "idle",
    job_id: null,
    current_epoch: 0,
    total_epochs: 0,
    latest_completed_epoch: 0,
    latest_metrics: null,
    history: [],
    best_validation_loss: null,
    cancellation_requested: false,
    checkpoint_file: "tiny_transformer_best.pt",
    checkpoint_available: false,
    error: null,
    model_config: null,
    training_config: null,
    ...overrides,
  };
}

export function createModelStatus(overrides: Partial<ModelStatus> = {}): ModelStatus {
  return {
    available: false,
    checkpoint_available: false,
    checkpoint_exists: false,
    checkpoint_file: "tiny_transformer_best.pt",
    metadata_file: "tiny_transformer_best.json",
    metadata: {},
    loaded: false,
    loaded_checkpoint: null,
    device: "cpu",
    architecture: "pre_norm_decoder",
    model_config: null,
    vocabulary_size: null,
    loaded_metadata: {},
    ...overrides,
  };
}

export function createPredictionResponse(overrides: Partial<PredictionResponse> = {}): PredictionResponse {
  return {
    input_text: "I love",
    tokens: ["<BOS>", "I", "love"],
    token_ids: [2, 7, 11],
    truncated: false,
    top_k: 2,
    temperature: 1,
    predictions: [
      { token: "you", token_id: 15, logit: 2.4, probability: 0.6 },
      { token: "music", token_id: 12, logit: 1.8, probability: 0.4 },
    ],
    probability_sum: 1,
    probability_label: "model probability",
    ...overrides,
  };
}

export function createPretrainedStatus(overrides: Partial<PretrainedStatus> = {}): PretrainedStatus {
  return {
    status: "not_loaded",
    loaded: false,
    loading: false,
    model_name: "sshleifer/tiny-gpt2",
    device: "cpu",
    dependencies_available: { torch: true, transformers: true },
    model: null,
    error: null,
    ...overrides,
  };
}

export interface MatchMediaController {
  setMatches: (matches: boolean) => void;
  mediaQuery: MediaQueryList;
}

export function installMatchMedia(initialMatches = false): MatchMediaController {
  let matches = initialMatches;
  const listeners = new Set<(event: MediaQueryListEvent) => void>();
  const eventListeners = new Set<EventListenerOrEventListenerObject>();
  const mediaQuery: MediaQueryList = {
    get matches() { return matches; },
    media: "(prefers-reduced-motion: reduce)",
    onchange: null,
    addEventListener: (_type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (listener) eventListeners.add(listener);
    },
    removeEventListener: (_type: string, listener: EventListenerOrEventListenerObject | null) => {
      if (listener) eventListeners.delete(listener);
    },
    addListener: (listener: ((this: MediaQueryList, event: MediaQueryListEvent) => void) | null) => {
      if (listener) listeners.add((event) => listener.call(mediaQuery, event));
    },
    removeListener: () => undefined,
    dispatchEvent: () => true,
  };
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: () => mediaQuery,
  });
  return {
    mediaQuery,
    setMatches(nextMatches) {
      matches = nextMatches;
      const event = new Event("change") as MediaQueryListEvent;
      Object.defineProperties(event, {
        matches: { value: matches },
        media: { value: mediaQuery.media },
      });
      listeners.forEach((listener) => listener(event));
      eventListeners.forEach((listener) => {
        if (typeof listener === "function") listener(event);
        else listener.handleEvent(event);
      });
    },
  };
}
