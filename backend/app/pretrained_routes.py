"""Thin FastAPI routes for the optional lazy pretrained-model lab."""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.pretrained_service import PretrainedServiceError, pretrained_service


router = APIRouter(prefix="/api/pretrained", tags=["pretrained model"])


class PretrainedLoadRequest(BaseModel):
    model_name: str | None = Field(default=None, max_length=200)


class PretrainedPredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1)
    temperature: float = Field(default=1.0, gt=0)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("text must not be empty")
        return text

    @field_validator("temperature")
    @classmethod
    def finite_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("temperature must be finite and greater than zero")
        return value


class PretrainedInspectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    layer: int = Field(default=0, ge=0)
    head: int = Field(default=0, ge=0)
    query_token: int | None = Field(default=None, ge=0)
    top_k: int = Field(default=5, ge=1)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("text must not be empty")
        return text


class PretrainedStatusResponse(BaseModel):
    status: str
    loaded: bool
    loading: bool
    model_name: str
    device: str
    dependencies_available: dict[str, bool]
    model: dict[str, object] | None
    error: str | None


class PretrainedPredictionItemResponse(BaseModel):
    token: str
    token_id: int
    logit: float
    probability: float


class PretrainedPredictionResponse(BaseModel):
    model_name: str
    device: str
    input_text: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int
    original_token_count: int
    context_truncated: bool
    temperature: float
    top_k: int
    predictions: list[PretrainedPredictionItemResponse]
    probability_sum: float
    top_probability_sum: float
    probability_label: str


class PretrainedConnectionResponse(BaseModel):
    key_index: int
    key_token: str
    key_token_id: int
    attention_weight: float
    is_future: bool


class PretrainedInspectResponse(BaseModel):
    model_name: str
    device: str
    input_text: str
    tokens: list[str]
    token_ids: list[int]
    token_count: int
    original_token_count: int
    context_truncated: bool
    selected_layer: int
    selected_head: int
    selected_query_index: int
    selected_query_token: str
    selected_query_token_id: int
    attention_shape: list[int]
    attention_matrix: list[list[float]]
    attention_row_sum: float
    connections: list[PretrainedConnectionResponse]
    top_predictions: list[PretrainedPredictionItemResponse]
    probability_sum: float
    attention_note: str


def _raise_safe_error(error: PretrainedServiceError) -> None:
    raise HTTPException(status_code=error.status_code, detail=error.as_detail()) from error


@router.get("/status", response_model=PretrainedStatusResponse)
def pretrained_status() -> dict[str, object]:
    return pretrained_service.status()


@router.post("/load", response_model=PretrainedStatusResponse)
def pretrained_load(payload: PretrainedLoadRequest) -> dict[str, object]:
    if payload.model_name and payload.model_name != pretrained_service.model_name:
        raise HTTPException(
            status_code=422,
            detail=(
                "The pretrained model is configured when the backend starts. Set "
                "PRETRAINED_MODEL_NAME and restart the backend to change it."
            ),
        )
    try:
        return pretrained_service.load()
    except PretrainedServiceError as error:
        _raise_safe_error(error)


@router.post("/inspect", response_model=PretrainedInspectResponse)
def pretrained_inspect(payload: PretrainedInspectRequest) -> dict[str, object]:
    try:
        return pretrained_service.inspect(
            payload.text,
            layer=payload.layer,
            head=payload.head,
            query_index=payload.query_token,
            top_k=payload.top_k,
        )
    except PretrainedServiceError as error:
        _raise_safe_error(error)


@router.post("/predict", response_model=PretrainedPredictionResponse)
def pretrained_predict(payload: PretrainedPredictRequest) -> dict[str, object]:
    try:
        return pretrained_service.predict(
            payload.text,
            top_k=payload.top_k,
            temperature=payload.temperature,
        )
    except PretrainedServiceError as error:
        _raise_safe_error(error)
