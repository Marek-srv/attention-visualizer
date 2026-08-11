import type {
  GenerationRequest,
  GenerationResponse,
  ModelStatus,
  PredictionRequest,
  PredictionResponse,
  PretrainedInspectRequest,
  PretrainedInspectResponse,
  PretrainedLoadRequest,
  PretrainedStatus,
  TinyInspectRequest,
  TinyInspectResponse,
  TrainingStartRequest,
  TrainingStatus,
} from "../types/api";

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload !== "object" || payload === null) return fallback;
  const detail = Reflect.get(payload, "detail");
  if (typeof detail === "string") return detail;
  if (typeof detail === "object" && detail !== null) {
    const nestedMessage = Reflect.get(detail, "message");
    if (typeof nestedMessage === "string") return nestedMessage;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => typeof item === "object" && item !== null ? Reflect.get(item, "msg") : null)
      .filter((item): item is string => typeof item === "string");
    if (messages.length) return messages.join(" · ");
  }
  const message = Reflect.get(payload, "message");
  return typeof message === "string" ? message : fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(payload, `Request failed (${response.status}).`));
  return payload as T;
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  trainingStatus: () => request<TrainingStatus>("/api/training/status"),
  startTraining: (body: TrainingStartRequest) => request<TrainingStatus>("/api/training/start", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  cancelTraining: () => request<TrainingStatus>("/api/training/cancel", { method: "POST", body: "{}" }),
  modelStatus: () => request<ModelStatus>("/api/model/status"),
  loadModel: () => request<ModelStatus>("/api/model/load", { method: "POST", body: "{}" }),
  predict: (body: PredictionRequest) => request<PredictionResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  generate: (body: GenerationRequest) => request<GenerationResponse>("/api/generate", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  inspectModel: (body: TinyInspectRequest) => request<TinyInspectResponse>("/api/model/inspect", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  pretrainedStatus: () => request<PretrainedStatus>("/api/pretrained/status"),
  loadPretrained: (body: PretrainedLoadRequest = {}) => request<PretrainedStatus>("/api/pretrained/load", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  inspectPretrained: (body: PretrainedInspectRequest) => request<PretrainedInspectResponse>("/api/pretrained/inspect", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  predictPretrained: (body: PredictionRequest) => request<PredictionResponse>("/api/pretrained/predict", {
    method: "POST",
    body: JSON.stringify(body),
  }),
};
