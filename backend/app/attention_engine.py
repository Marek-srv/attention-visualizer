"""Transparent scaled dot-product attention calculations for Phase 4."""

from __future__ import annotations

import math
from typing import Any


def dot_product(left: list[float], right: list[float]) -> float:
    """Return the scalar dot product of two equal-length vectors."""
    return sum(left[index] * right[index] for index in range(len(left)))


def calculate_raw_scores(queries: list[list[float]], keys: list[list[float]]) -> list[list[float]]:
    return [[round(dot_product(query, key), 4) for key in keys] for query in queries]


def scale_scores(raw_scores: list[list[float]], key_dimension: int) -> list[list[float]]:
    scale_factor = math.sqrt(key_dimension)
    return [[round(score / scale_factor, 4) for score in row] for row in raw_scores]


def generate_causal_mask(token_count: int) -> list[list[bool]]:
    """Use True for visible keys and False for masked future keys."""
    return [[key_position <= query_position for key_position in range(token_count)] for query_position in range(token_count)]


def apply_causal_mask(scores: list[list[float]], mask: list[list[bool]]) -> list[list[float | None]]:
    return [[score if mask[row][column] else None for column, score in enumerate(scores[row])] for row in range(len(scores))]


def stable_softmax(values: list[float]) -> list[float]:
    """Compute softmax after subtracting the maximum to prevent overflow."""
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def calculate_attention_weights(masked_scores: list[list[float | None]]) -> list[list[float]]:
    weights: list[list[float]] = []
    for row in masked_scores:
        allowed_values = [value for value in row if value is not None]
        allowed_weights = iter(stable_softmax(allowed_values))
        weights.append([round(next(allowed_weights), 4) if value is not None else 0.0 for value in row])
    return weights


def calculate_context_vectors(attention_weights: list[list[float]], values: list[list[float]]) -> list[list[float]]:
    if not values:
        return []
    dimension = len(values[0])
    return [
        [round(sum(row[key] * values[key][output] for key in range(len(values))), 4) for output in range(dimension)]
        for row in attention_weights
    ]


def generate_pair_calculations(
    tokens: list[str],
    queries: list[list[float]],
    keys: list[list[float]],
    raw_scores: list[list[float]],
    scaled_scores: list[list[float]],
    mask: list[list[bool]],
    attention_weights: list[list[float]],
) -> list[dict[str, Any]]:
    calculations: list[dict[str, Any]] = []
    for query_position, query in enumerate(queries):
        for key_position, key in enumerate(keys):
            products = [
                {
                    "dimension": dimension,
                    "query_value": query[dimension],
                    "key_value": key[dimension],
                    "product": round(query[dimension] * key[dimension], 4),
                }
                for dimension in range(len(query))
            ]
            calculations.append(
                {
                    "query_token": tokens[query_position],
                    "query_position": query_position,
                    "key_token": tokens[key_position],
                    "key_position": key_position,
                    "causally_masked": not mask[query_position][key_position],
                    "query_vector": query,
                    "key_vector": key,
                    "products": products,
                    "raw_score": raw_scores[query_position][key_position],
                    "scaled_score": scaled_scores[query_position][key_position],
                    "attention_weight": attention_weights[query_position][key_position],
                }
            )
    return calculations


def generate_context_calculations(
    tokens: list[str], attention_weights: list[list[float]], values: list[list[float]], context_vectors: list[list[float]]
) -> list[dict[str, Any]]:
    calculations: list[dict[str, Any]] = []
    if not values:
        return calculations
    for query_position, query_token in enumerate(tokens):
        dimensions = []
        for output_dimension in range(len(values[0])):
            terms = [
                {
                    "key_token": tokens[key_position],
                    "key_position": key_position,
                    "attention_weight": attention_weights[query_position][key_position],
                    "value": values[key_position][output_dimension],
                    "product": round(attention_weights[query_position][key_position] * values[key_position][output_dimension], 4),
                }
                for key_position in range(len(tokens))
            ]
            dimensions.append(
                {
                    "output_dimension": output_dimension,
                    "terms": terms,
                    "result": context_vectors[query_position][output_dimension],
                }
            )
        calculations.append(
            {
                "query_token": query_token,
                "query_position": query_position,
                "dimensions": dimensions,
            }
        )
    return calculations


def calculate_attention(tokens: list[str], projections: list[dict[str, Any]], key_dimension: int = 4) -> dict[str, Any]:
    queries = [projection["query"] for projection in projections]
    keys = [projection["key"] for projection in projections]
    values = [projection["value"] for projection in projections]
    raw_scores = calculate_raw_scores(queries, keys)
    scaled_scores = scale_scores(raw_scores, key_dimension)
    causal_mask = generate_causal_mask(len(tokens))
    masked_scores = apply_causal_mask(scaled_scores, causal_mask)
    attention_weights = calculate_attention_weights(masked_scores)
    context_vectors = calculate_context_vectors(attention_weights, values)
    return {
        "key_dimension": key_dimension,
        "scale_factor": math.sqrt(key_dimension),
        "causal_mask": causal_mask,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "masked_scores": masked_scores,
        "attention_weights": attention_weights,
        "context_vectors": context_vectors,
        "row_sums": [round(sum(row), 4) for row in attention_weights],
        "calculations": generate_pair_calculations(
            tokens, queries, keys, raw_scores, scaled_scores, causal_mask, attention_weights
        ),
        "context_calculations": generate_context_calculations(tokens, attention_weights, values, context_vectors),
    }
