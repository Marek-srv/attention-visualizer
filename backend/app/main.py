from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import math
import os
from numbers import Real
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.embedding_engine import EMBEDDING_DIMENSION, VOCABULARY, embed_text
from app.attention_engine import calculate_attention
from app.projection_engine import DEFAULT_WK, DEFAULT_WQ, DEFAULT_WV, Matrix, project_vector
from app.multi_head_engine import DEFAULT_WO, calculate_multi_head_attention
from app.normalization_engine import (
    DEFAULT_BETA,
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    calculate_attention_sublayer,
)
from app.feed_forward_engine import (
    DEFAULT_B1,
    DEFAULT_B2,
    DEFAULT_LN2_BETA,
    DEFAULT_LN2_GAMMA,
    DEFAULT_W1,
    DEFAULT_W2,
    calculate_feed_forward_sublayer,
)
from app.pretrained_routes import router as pretrained_router
from app.trainable_routes import router as trainable_router


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
)


def get_cors_origins() -> list[str]:
    """Return local defaults plus unique, explicitly configured deployment origins."""
    configured_origins = (origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(","))
    return list(
        dict.fromkeys(
            (*DEFAULT_CORS_ORIGINS, *(origin for origin in configured_origins if origin))
        )
    )


def configure_cors(application: FastAPI) -> None:
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


app = FastAPI(
    title="Transformer Attention Visualizer API",
    version="1.0.0",
    description=(
        "Transparent Phase 1–7 toy calculations plus local tiny-model training, "
        "prediction, traced inspection, and optional pretrained-model inspection."
    ),
)

configure_cors(app)
app.include_router(pretrained_router)
app.include_router(trainable_router)


def _json_safe_validation_detail(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Exception):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe_validation_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_detail(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Any, exception: RequestValidationError) -> JSONResponse:
    """Ensure even rejected NaN/Infinity inputs produce valid JSON."""
    return JSONResponse(status_code=422, content={"detail": _json_safe_validation_detail(exception.errors())})


class ProjectionWeights(BaseModel):
    query: Matrix
    key: Matrix
    value: Matrix
    output: Matrix | None = None

    @field_validator("query", "key", "value", "output", mode="before")
    @classmethod
    def validate_matrix(cls, matrix: Any) -> Any:
        if matrix is None:
            return matrix
        if not isinstance(matrix, list) or len(matrix) != EMBEDDING_DIMENSION:
            raise ValueError("matrix must contain exactly 4 rows")
        for row in matrix:
            if not isinstance(row, list) or len(row) != EMBEDDING_DIMENSION:
                raise ValueError("each matrix row must contain exactly 4 numeric values")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("each matrix value must be numeric")
                if not math.isfinite(float(value)):
                    raise ValueError("matrix values must be finite (not NaN or infinite)")
        return matrix


class NormalizationParameters(BaseModel):
    gamma: list[float] = Field(default_factory=lambda: list(DEFAULT_GAMMA))
    beta: list[float] = Field(default_factory=lambda: list(DEFAULT_BETA))
    epsilon: float = DEFAULT_EPSILON

    @field_validator("gamma", "beta", mode="before")
    @classmethod
    def validate_vector(cls, vector: Any) -> Any:
        if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
            raise ValueError("normalization vector must contain exactly 4 finite numeric values")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("normalization values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("normalization values must be finite (not NaN or infinite)")
        return vector

    @field_validator("epsilon", mode="before")
    @classmethod
    def validate_epsilon(cls, epsilon: Any) -> Any:
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise ValueError("epsilon must be a finite number greater than zero")
        if not math.isfinite(float(epsilon)) or epsilon <= 0:
            raise ValueError("epsilon must be a finite number greater than zero")
        return epsilon


class FeedForwardParameters(BaseModel):
    input_weights: list[list[float]] = Field(default_factory=lambda: [list(row) for row in DEFAULT_W1])
    input_bias: list[float] = Field(default_factory=lambda: list(DEFAULT_B1))
    output_weights: list[list[float]] = Field(default_factory=lambda: [list(row) for row in DEFAULT_W2])
    output_bias: list[float] = Field(default_factory=lambda: list(DEFAULT_B2))
    normalization: NormalizationParameters = Field(
        default_factory=lambda: NormalizationParameters(gamma=DEFAULT_LN2_GAMMA, beta=DEFAULT_LN2_BETA)
    )

    @field_validator("input_weights", mode="before")
    @classmethod
    def validate_input_weights(cls, matrix: Any) -> Any:
        return cls._validate_matrix(matrix, 4, 8, "W1")

    @field_validator("output_weights", mode="before")
    @classmethod
    def validate_output_weights(cls, matrix: Any) -> Any:
        return cls._validate_matrix(matrix, 8, 4, "W2")

    @field_validator("input_bias", mode="before")
    @classmethod
    def validate_input_bias(cls, vector: Any) -> Any:
        return cls._validate_vector(vector, 8, "b1")

    @field_validator("output_bias", mode="before")
    @classmethod
    def validate_output_bias(cls, vector: Any) -> Any:
        return cls._validate_vector(vector, 4, "b2")

    @staticmethod
    def _validate_matrix(matrix: Any, rows: int, columns: int, name: str) -> Any:
        if not isinstance(matrix, list) or len(matrix) != rows:
            raise ValueError(f"{name} must contain exactly {rows} rows")
        for row in matrix:
            if not isinstance(row, list) or len(row) != columns:
                raise ValueError(f"each {name} row must contain exactly {columns} finite numeric values")
            FeedForwardParameters._validate_finite_values(row, name)
        return matrix

    @staticmethod
    def _validate_vector(vector: Any, length: int, name: str) -> Any:
        if not isinstance(vector, list) or len(vector) != length:
            raise ValueError(f"{name} must contain exactly {length} finite numeric values")
        FeedForwardParameters._validate_finite_values(vector, name)
        return vector

    @staticmethod
    def _validate_finite_values(values: list[Any], name: str) -> None:
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} values must be finite numbers")


class InspectRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    weights: ProjectionWeights | None = None
    normalization: NormalizationParameters | None = None
    feed_forward: FeedForwardParameters | None = None


class TokenResult(BaseModel):
    token: str
    normalized: str
    token_id: int
    position: int
    token_embedding: list[float]
    position_embedding: list[float]
    combined_embedding: list[float]


class CalculationTerm(BaseModel):
    input_dimension: int
    input_value: float
    weight_value: float
    product: float


class DimensionBreakdown(BaseModel):
    output_dimension: int
    terms: list[CalculationTerm]
    result: float


class ProjectionBreakdown(BaseModel):
    query: list[DimensionBreakdown]
    key: list[DimensionBreakdown]
    value: list[DimensionBreakdown]


class ProjectionResult(BaseModel):
    token: str
    position: int
    input_vector: list[float]
    query: list[float]
    key: list[float]
    value: list[float]
    breakdown: ProjectionBreakdown


class AttentionProduct(BaseModel):
    dimension: int
    query_value: float
    key_value: float
    product: float


class AttentionCalculation(BaseModel):
    query_token: str
    query_position: int
    key_token: str
    key_position: int
    causally_masked: bool
    query_vector: list[float]
    key_vector: list[float]
    products: list[AttentionProduct]
    raw_score: float
    scaled_score: float
    attention_weight: float


class ContextTerm(BaseModel):
    key_token: str
    key_position: int
    attention_weight: float
    value: float
    product: float


class ContextDimensionCalculation(BaseModel):
    output_dimension: int
    terms: list[ContextTerm]
    result: float


class ContextCalculation(BaseModel):
    query_token: str
    query_position: int
    dimensions: list[ContextDimensionCalculation]


class AttentionResult(BaseModel):
    key_dimension: int
    scale_factor: float
    causal_mask: list[list[bool]]
    raw_scores: list[list[float]]
    scaled_scores: list[list[float]]
    masked_scores: list[list[float | None]]
    attention_weights: list[list[float]]
    context_vectors: list[list[float]]
    row_sums: list[float]
    calculations: list[AttentionCalculation]
    context_calculations: list[ContextCalculation]


class HeadAttentionResult(BaseModel):
    head_index: int
    dimension_indices: list[int]
    query_vectors: list[list[float]]
    key_vectors: list[list[float]]
    value_vectors: list[list[float]]
    raw_scores: list[list[float]]
    scaled_scores: list[list[float]]
    causal_mask: list[list[bool]]
    masked_scores: list[list[float | None]]
    attention_weights: list[list[float]]
    context_vectors: list[list[float]]
    row_sums: list[float]
    calculations: list[AttentionCalculation]


class OutputProjectionTerm(BaseModel):
    input_dimension: int
    concatenated_input_value: float
    weight_value: float
    product: float


class OutputDimensionCalculation(BaseModel):
    output_dimension: int
    terms: list[OutputProjectionTerm]
    result: float


class OutputCalculation(BaseModel):
    token: str
    position: int
    dimensions: list[OutputDimensionCalculation]


class MultiHeadAttentionResult(BaseModel):
    model_dimension: int
    number_of_heads: int
    head_dimension: int
    scale_factor: float
    heads: list[HeadAttentionResult]
    concatenated_contexts: list[list[float]]
    output_weight_matrix: list[list[float]]
    projected_outputs: list[list[float]]
    output_calculations: list[OutputCalculation]


class ResidualTerm(BaseModel):
    dimension: int
    input_value: float
    attention_value: float
    residual_value: float


class NormalizationTerm(BaseModel):
    dimension: int
    residual_value: float
    mean: float
    centered_value: float
    standard_deviation: float
    normalized_value: float
    gamma: float
    beta: float
    output_value: float


class NormalizationCalculation(BaseModel):
    token: str
    position: int
    input_vector: list[float]
    attention_output: list[float]
    residual_terms: list[ResidualTerm]
    residual_vector: list[float]
    mean: float
    variance: float
    epsilon: float
    standard_deviation: float
    normalization_terms: list[NormalizationTerm]
    normalized_vector: list[float]
    layer_norm_output: list[float]


class AttentionSublayerResult(BaseModel):
    architecture: str
    input_vectors: list[list[float]]
    attention_outputs: list[list[float]]
    residual_vectors: list[list[float]]
    gamma: list[float]
    beta: list[float]
    epsilon: float
    normalized_vectors: list[list[float]]
    layer_norm_outputs: list[list[float]]
    calculations: list[NormalizationCalculation]


class FeedForwardSublayerResult(BaseModel):
    input_dimension: int
    hidden_dimension: int
    activation: str
    input_vectors: list[list[float]]
    input_weight_matrix: list[list[float]]
    input_bias: list[float]
    pre_activation_vectors: list[list[float]]
    activated_vectors: list[list[float]]
    output_weight_matrix: list[list[float]]
    output_bias: list[float]
    feed_forward_outputs: list[list[float]]
    residual_vectors: list[list[float]]
    normalization: NormalizationParameters
    normalized_vectors: list[list[float]]
    transformer_block_outputs: list[list[float]]
    calculations: list[dict[str, Any]]


class InspectResponse(BaseModel):
    text: str
    character_count: int
    phase: int
    token_count: int
    vocabulary_size: int
    embedding_dimension: int
    tokens: list[TokenResult]
    weights: ProjectionWeights
    projections: list[ProjectionResult]
    attention: AttentionResult
    multi_head_attention: MultiHeadAttentionResult
    attention_sublayer: AttentionSublayerResult
    feed_forward_sublayer: FeedForwardSublayerResult


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "attention-visualizer-api"}


@app.post("/api/inspect", response_model=InspectResponse)
def inspect_text(payload: InspectRequest) -> InspectResponse:
    cleaned_text = payload.text.strip()
    if not cleaned_text:
        cleaned_text = payload.text

    tokens = embed_text(cleaned_text)
    supplied_weights = payload.weights or ProjectionWeights(
        query=DEFAULT_WQ,
        key=DEFAULT_WK,
        value=DEFAULT_WV,
    )
    weights = ProjectionWeights(
        query=supplied_weights.query,
        key=supplied_weights.key,
        value=supplied_weights.value,
        output=supplied_weights.output or DEFAULT_WO,
    )
    weight_data = weights.model_dump()
    projections = []
    for token in tokens:
        projected = project_vector(token.combined_embedding, weight_data)
        projections.append(
            ProjectionResult(
                token=token.token,
                position=token.position,
                input_vector=token.combined_embedding,
                **projected,
            )
        )
    projection_data = [projection.model_dump() for projection in projections]
    attention = calculate_attention([token.token for token in tokens], projection_data)
    multi_head_attention = calculate_multi_head_attention(
        [token.token for token in tokens], projection_data, weights.output or DEFAULT_WO
    )
    normalization = payload.normalization or NormalizationParameters()
    attention_sublayer = calculate_attention_sublayer(
        [token.token for token in tokens],
        [token.combined_embedding for token in tokens],
        multi_head_attention["projected_outputs"],
        normalization.gamma,
        normalization.beta,
        normalization.epsilon,
    )
    feed_forward_parameters = payload.feed_forward or FeedForwardParameters()
    feed_forward_sublayer = calculate_feed_forward_sublayer(
        [token.token for token in tokens],
        attention_sublayer["layer_norm_outputs"],
        feed_forward_parameters.input_weights,
        feed_forward_parameters.input_bias,
        feed_forward_parameters.output_weights,
        feed_forward_parameters.output_bias,
        feed_forward_parameters.normalization.gamma,
        feed_forward_parameters.normalization.beta,
        feed_forward_parameters.normalization.epsilon,
    )
    return InspectResponse(
        text=cleaned_text,
        character_count=len(cleaned_text),
        phase=7,
        token_count=len(tokens),
        vocabulary_size=len(VOCABULARY),
        embedding_dimension=EMBEDDING_DIMENSION,
        tokens=[TokenResult(**vars(token)) for token in tokens],
        weights=weights,
        projections=projections,
        attention=AttentionResult(**attention),
        multi_head_attention=MultiHeadAttentionResult(**multi_head_attention),
        attention_sublayer=AttentionSublayerResult(**attention_sublayer),
        feed_forward_sublayer=FeedForwardSublayerResult(**feed_forward_sublayer),
    )
