"""Thin FastAPI routes for local tiny-model training and inference."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.training.checkpoint import CheckpointError, CheckpointManager, CheckpointNotFoundError
from app.training.inference import InferenceValidationError, ModelNotLoadedError, TinyModelService
from app.training.config import ModelConfig, TrainingConfig
from app.training.trainer import TrainingAlreadyRunningError, TrainingManager


router = APIRouter(tags=["trainable tiny model"])
LOGGER = logging.getLogger(__name__)
checkpoint_manager = CheckpointManager()
model_service = TinyModelService(checkpoint_manager)
training_manager = TrainingManager(checkpoint_manager=checkpoint_manager)


class ModelConfigurationRequest(BaseModel):
    context_length: int = Field(default=16, ge=2, le=512)
    d_model: int = Field(default=32, ge=4, le=1024)
    number_of_heads: int = Field(default=4, ge=1, le=64)
    number_of_layers: int = Field(default=2, ge=1, le=24)
    feed_forward_dimension: int = Field(default=64, ge=4, le=8192)
    dropout: float = Field(default=0.1, ge=0, lt=1)

    @field_validator("dropout")
    @classmethod
    def finite_dropout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("dropout must be finite")
        return value


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    epochs: int = Field(default=100, ge=1, le=500)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.003, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=10)
    gradient_clip: float = Field(default=1.0, gt=0, le=100)
    seed: int = Field(default=42, ge=-(2**31), le=2**31 - 1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    model_settings: ModelConfigurationRequest | None = Field(default=None, alias="model_config")

    @field_validator("learning_rate", "weight_decay", "gradient_clip", "validation_fraction")
    @classmethod
    def finite_training_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("training parameters must be finite")
        return value


class ModelLoadRequest(BaseModel):
    checkpoint_name: str = Field(default="tiny_transformer_best.pt", min_length=1, max_length=100)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1)
    temperature: float = Field(default=1.0, gt=0)

    @field_validator("text")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def finite_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("temperature must be finite and greater than zero")
        return value


class GenerationRequest(PredictionRequest):
    max_new_tokens: int = Field(default=8, ge=1, le=50)
    strategy: Literal["greedy", "sample"] = "greedy"
    seed: int = Field(default=42, ge=-(2**31), le=2**31 - 1)


class ModelInspectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)
    query_token: int | None = Field(default=None, ge=0)
    key_token: int | None = Field(default=None, ge=0)
    hidden_dimension: int = Field(default=0, ge=0)
    top_k: int = Field(default=5, ge=1)

    @field_validator("text")
    @classmethod
    def non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


class TrainingMetricResponse(BaseModel):
    epoch: int
    training_loss: float
    validation_loss: float
    training_perplexity: float
    validation_perplexity: float
    learning_rate: float
    duration_seconds: float


class TrainingStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    state: Literal["idle", "running", "completed", "failed", "cancelled"]
    status: Literal["idle", "running", "completed", "failed", "cancelled"]
    job_id: str | None
    current_epoch: int
    total_epochs: int
    latest_completed_epoch: int
    latest_metrics: TrainingMetricResponse | None
    history: list[TrainingMetricResponse]
    best_validation_loss: float | None
    cancellation_requested: bool
    checkpoint_file: str
    checkpoint_available: bool
    error: str | None
    model_details: dict[str, Any] | None = Field(default=None, alias="model_config")
    training_config: dict[str, Any] | None = None


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    available: bool
    checkpoint_available: bool
    checkpoint_exists: bool
    checkpoint_file: str
    metadata_file: str
    metadata: dict[str, Any]
    loaded: bool
    loaded_checkpoint: str | None
    device: str
    architecture: str
    model_details: dict[str, Any] | None = Field(alias="model_config")
    vocabulary_size: int | None
    loaded_metadata: dict[str, Any]


class PredictionItemResponse(BaseModel):
    token: str
    token_id: int
    logit: float
    probability: float


class PredictionResponse(BaseModel):
    input_text: str
    tokens: list[str]
    token_ids: list[int]
    truncated: bool
    top_k: int
    temperature: float
    predictions: list[PredictionItemResponse]
    probability_sum: float
    probability_label: str


class GenerationStepResponse(BaseModel):
    step: int
    chosen_token: str
    chosen_token_id: int
    chosen_probability: float
    top_predictions: list[PredictionItemResponse]
    is_eos: bool


class GenerationResponse(BaseModel):
    input_text: str
    input_tokens: list[str]
    input_token_ids: list[int]
    truncated: bool
    generated_text: str
    generated_tokens: list[str]
    generated_token_ids: list[int]
    strategy: Literal["greedy", "sample"]
    seed: int
    temperature: float
    top_k: int
    max_new_tokens: int
    stop_reason: Literal["eos", "max_new_tokens"]
    steps: list[GenerationStepResponse]
    probability_label: str


class ModelInspectResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    input_text: str
    tokens: list[str]
    token_ids: list[int]
    truncated: bool
    architecture: str
    attention_note: str
    model_details: dict[str, Any] = Field(alias="model_config")
    selection: dict[str, int]
    shapes: dict[str, list[int]]
    token_embeddings: list[list[float]]
    position_embeddings: list[list[float]]
    combined_embeddings: list[list[float]]
    layer_trace: dict[str, Any]
    selected_attention_calculation: dict[str, Any]
    token_connections: list[dict[str, Any]]
    selected_hidden_values: dict[str, Any]
    final_hidden_states: list[list[float]]
    vocabulary_logits: list[list[float]]
    vocabulary_probabilities: list[list[float]]
    top_predictions: list[PredictionItemResponse]
    probability_sum: float


def _run_inference(call):
    try:
        return call()
    except InferenceValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelNotLoadedError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except CheckpointNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (CheckpointError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("Local tiny-model inference failed")
        raise HTTPException(status_code=500, detail="The local model operation failed safely.") from error


@router.get("/api/training/status", response_model=TrainingStatusResponse)
def training_status() -> dict[str, object]:
    return training_manager.status()


@router.post("/api/training/start", response_model=TrainingStatusResponse)
def training_start(payload: TrainingStartRequest) -> dict[str, object]:
    try:
        model_values = payload.model_settings.model_dump() if payload.model_settings else {}
        model_config = ModelConfig(**model_values)
        training_config = TrainingConfig(
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
            weight_decay=payload.weight_decay,
            gradient_clip=payload.gradient_clip,
            seed=payload.seed,
            validation_fraction=payload.validation_fraction,
        )
        return training_manager.start(model_config, training_config)
    except TrainingAlreadyRunningError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@router.post("/api/training/cancel", response_model=TrainingStatusResponse)
def training_cancel() -> dict[str, object]:
    return training_manager.cancel()


@router.get("/api/model/status", response_model=ModelStatusResponse)
def model_status() -> dict[str, object]:
    status = model_service.status()
    available = bool(status.get("available"))
    return {**status, "checkpoint_available": available, "checkpoint_exists": available}


@router.post("/api/model/load", response_model=ModelStatusResponse)
def model_load(payload: ModelLoadRequest) -> dict[str, object]:
    loaded = _run_inference(lambda: model_service.load(payload.checkpoint_name))
    available = bool(loaded.get("available"))
    return {**loaded, "checkpoint_available": available, "checkpoint_exists": available}


@router.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> dict[str, object]:
    return _run_inference(
        lambda: model_service.predict(payload.text, top_k=payload.top_k, temperature=payload.temperature)
    )


@router.post("/api/generate", response_model=GenerationResponse)
def generate(payload: GenerationRequest) -> dict[str, object]:
    return _run_inference(
        lambda: model_service.generate(
            payload.text,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            strategy=payload.strategy,
            seed=payload.seed,
        )
    )


@router.post("/api/model/inspect", response_model=ModelInspectResponse)
def inspect_model(payload: ModelInspectRequest) -> dict[str, object]:
    return _run_inference(
        lambda: model_service.inspect(
            payload.text,
            layer=payload.layer,
            head=payload.head,
            query_token=payload.query_token,
            key_token=payload.key_token,
            hidden_dimension=payload.hidden_dimension,
            top_k=payload.top_k,
        )
    )
