import type {
  GenerationRequest,
  GenerationResponse,
  HealthResponse,
  ModelLoadRequest,
  ModelStatus,
  PredictionRequest,
  PredictionResponse,
  PretrainedInspectRequest,
  PretrainedInspectResponse,
  PretrainedLoadRequest,
  PretrainedPredictionResponse,
  PretrainedStatus,
  TinyInspectRequest,
  TinyInspectResponse,
  ToyInspectRequest,
  ToyInspectResponse,
  TrainingStartRequest,
  TrainingStatus,
} from "../types/api";
import { apiClient, type ApiRequestOptions } from "./client";

export const API_ENDPOINTS = {
  health: "/api/health",
  inspectToy: "/api/inspect",
  trainingStatus: "/api/training/status",
  startTraining: "/api/training/start",
  cancelTraining: "/api/training/cancel",
  modelStatus: "/api/model/status",
  loadModel: "/api/model/load",
  predict: "/api/predict",
  generate: "/api/generate",
  inspectModel: "/api/model/inspect",
  pretrainedStatus: "/api/pretrained/status",
  loadPretrained: "/api/pretrained/load",
  inspectPretrained: "/api/pretrained/inspect",
  predictPretrained: "/api/pretrained/predict",
} as const;

export interface JsonApiClient {
  get<TResponse>(path: string, options?: ApiRequestOptions): Promise<TResponse>;
  post<TResponse, TBody>(path: string, body: TBody, options?: ApiRequestOptions): Promise<TResponse>;
}

export function createAttentionApi(client: JsonApiClient) {
  return {
    health: (options: ApiRequestOptions = {}) =>
      client.get<HealthResponse>(API_ENDPOINTS.health, options),
    inspectToy: (payload: ToyInspectRequest, options: ApiRequestOptions = {}) =>
      client.post<ToyInspectResponse, ToyInspectRequest>(API_ENDPOINTS.inspectToy, payload, options),
    trainingStatus: (options: ApiRequestOptions = {}) =>
      client.get<TrainingStatus>(API_ENDPOINTS.trainingStatus, options),
    startTraining: (payload: TrainingStartRequest, options: ApiRequestOptions = {}) =>
      client.post<TrainingStatus, TrainingStartRequest>(API_ENDPOINTS.startTraining, payload, options),
    cancelTraining: (options: ApiRequestOptions = {}) =>
      client.post<TrainingStatus, Record<string, never>>(API_ENDPOINTS.cancelTraining, {}, options),
    modelStatus: (options: ApiRequestOptions = {}) =>
      client.get<ModelStatus>(API_ENDPOINTS.modelStatus, options),
    loadModel: (payload: ModelLoadRequest = {}, options: ApiRequestOptions = {}) =>
      client.post<ModelStatus, ModelLoadRequest>(API_ENDPOINTS.loadModel, payload, options),
    predict: (payload: PredictionRequest, options: ApiRequestOptions = {}) =>
      client.post<PredictionResponse, PredictionRequest>(API_ENDPOINTS.predict, payload, options),
    generate: (payload: GenerationRequest, options: ApiRequestOptions = {}) =>
      client.post<GenerationResponse, GenerationRequest>(API_ENDPOINTS.generate, payload, options),
    inspectModel: (payload: TinyInspectRequest, options: ApiRequestOptions = {}) =>
      client.post<TinyInspectResponse, TinyInspectRequest>(API_ENDPOINTS.inspectModel, payload, options),
    pretrainedStatus: (options: ApiRequestOptions = {}) =>
      client.get<PretrainedStatus>(API_ENDPOINTS.pretrainedStatus, options),
    loadPretrained: (payload: PretrainedLoadRequest = {}, options: ApiRequestOptions = {}) =>
      client.post<PretrainedStatus, PretrainedLoadRequest>(API_ENDPOINTS.loadPretrained, payload, options),
    inspectPretrained: (payload: PretrainedInspectRequest, options: ApiRequestOptions = {}) =>
      client.post<PretrainedInspectResponse, PretrainedInspectRequest>(API_ENDPOINTS.inspectPretrained, payload, options),
    predictPretrained: (payload: PredictionRequest, options: ApiRequestOptions = {}) =>
      client.post<PretrainedPredictionResponse, PredictionRequest>(API_ENDPOINTS.predictPretrained, payload, options),
  };
}

export const api = createAttentionApi(apiClient);
