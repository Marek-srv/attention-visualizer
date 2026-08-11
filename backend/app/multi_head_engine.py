"""Transparent two-head causal self-attention for Phase 5."""

from __future__ import annotations

import math
from typing import Any

from app.attention_engine import (
    apply_causal_mask,
    calculate_attention_weights,
    calculate_context_vectors,
    calculate_raw_scores,
    generate_causal_mask,
    generate_pair_calculations,
    scale_scores,
)


MODEL_DIMENSION = 4
NUMBER_OF_HEADS = 2
HEAD_DIMENSION = MODEL_DIMENSION // NUMBER_OF_HEADS

if MODEL_DIMENSION % NUMBER_OF_HEADS != 0:
    raise ValueError("MODEL_DIMENSION must be divisible by NUMBER_OF_HEADS")

DEFAULT_WO: list[list[float]] = [
    [0.60, -0.20, 0.30, 0.10],
    [0.20, 0.70, -0.10, 0.40],
    [-0.30, 0.20, 0.80, -0.20],
    [0.40, 0.10, 0.20, 0.60],
]


def split_vectors_into_heads(vectors: list[list[float]]) -> list[list[list[float]]]:
    """Return heads × tokens × head dimensions."""
    return [
        [vector[head * HEAD_DIMENSION : (head + 1) * HEAD_DIMENSION] for vector in vectors]
        for head in range(NUMBER_OF_HEADS)
    ]


def calculate_head(
    head_index: int,
    tokens: list[str],
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> dict[str, Any]:
    raw_scores = calculate_raw_scores(queries, keys)
    scaled_scores = scale_scores(raw_scores, HEAD_DIMENSION)
    causal_mask = generate_causal_mask(len(tokens))
    masked_scores = apply_causal_mask(scaled_scores, causal_mask)
    attention_weights = calculate_attention_weights(masked_scores)
    context_vectors = calculate_context_vectors(attention_weights, values)
    return {
        "head_index": head_index,
        "dimension_indices": list(range(head_index * HEAD_DIMENSION, (head_index + 1) * HEAD_DIMENSION)),
        "query_vectors": queries,
        "key_vectors": keys,
        "value_vectors": values,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "causal_mask": causal_mask,
        "masked_scores": masked_scores,
        "attention_weights": attention_weights,
        "context_vectors": context_vectors,
        "row_sums": [round(sum(row), 4) for row in attention_weights],
        "calculations": generate_pair_calculations(
            tokens, queries, keys, raw_scores, scaled_scores, causal_mask, attention_weights
        ),
    }


def concatenate_head_contexts(heads: list[dict[str, Any]], token_count: int) -> list[list[float]]:
    return [
        [value for head in heads for value in head["context_vectors"][token_position]]
        for token_position in range(token_count)
    ]


def apply_output_projection(
    concatenated_contexts: list[list[float]], output_weights: list[list[float]], tokens: list[str]
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    outputs: list[list[float]] = []
    calculations: list[dict[str, Any]] = []
    for token_position, context in enumerate(concatenated_contexts):
        output_vector: list[float] = []
        dimensions = []
        for output_dimension in range(MODEL_DIMENSION):
            terms = []
            total = 0.0
            for input_dimension in range(MODEL_DIMENSION):
                product = context[input_dimension] * output_weights[input_dimension][output_dimension]
                total += product
                terms.append(
                    {
                        "input_dimension": input_dimension,
                        "concatenated_input_value": context[input_dimension],
                        "weight_value": output_weights[input_dimension][output_dimension],
                        "product": round(product, 4),
                    }
                )
            result = round(total, 4)
            output_vector.append(result)
            dimensions.append({"output_dimension": output_dimension, "terms": terms, "result": result})
        outputs.append(output_vector)
        calculations.append(
            {"token": tokens[token_position], "position": token_position, "dimensions": dimensions}
        )
    return outputs, calculations


def calculate_multi_head_attention(
    tokens: list[str], projections: list[dict[str, Any]], output_weights: list[list[float]]
) -> dict[str, Any]:
    split_queries = split_vectors_into_heads([projection["query"] for projection in projections])
    split_keys = split_vectors_into_heads([projection["key"] for projection in projections])
    split_values = split_vectors_into_heads([projection["value"] for projection in projections])
    heads = [
        calculate_head(head, tokens, split_queries[head], split_keys[head], split_values[head])
        for head in range(NUMBER_OF_HEADS)
    ]
    concatenated_contexts = concatenate_head_contexts(heads, len(tokens))
    projected_outputs, output_calculations = apply_output_projection(
        concatenated_contexts, output_weights, tokens
    )
    return {
        "model_dimension": MODEL_DIMENSION,
        "number_of_heads": NUMBER_OF_HEADS,
        "head_dimension": HEAD_DIMENSION,
        "scale_factor": round(math.sqrt(HEAD_DIMENSION), 4),
        "heads": heads,
        "concatenated_contexts": concatenated_contexts,
        "output_weight_matrix": output_weights,
        "projected_outputs": projected_outputs,
        "output_calculations": output_calculations,
    }
