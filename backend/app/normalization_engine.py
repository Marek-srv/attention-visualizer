"""Transparent residual connection and layer normalization for Phase 6."""

from __future__ import annotations

import math
from typing import Any


MODEL_DIMENSION = 4
DEFAULT_GAMMA = [1.0, 1.0, 1.0, 1.0]
DEFAULT_BETA = [0.0, 0.0, 0.0, 0.0]
DEFAULT_EPSILON = 0.00001


def add_vectors(left: list[float], right: list[float]) -> list[float]:
    return [left[index] + right[index] for index in range(len(left))]


def calculate_mean(vector: list[float]) -> float:
    return sum(vector) / len(vector)


def calculate_population_variance(vector: list[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in vector) / len(vector)


def calculate_standard_deviation(variance: float, epsilon: float) -> float:
    return math.sqrt(variance + epsilon)


def normalize_vector(vector: list[float], mean: float, standard_deviation: float) -> list[float]:
    return [(value - mean) / standard_deviation for value in vector]


def apply_gamma(normalized: list[float], gamma: list[float]) -> list[float]:
    return [normalized[index] * gamma[index] for index in range(len(normalized))]


def apply_beta(scaled: list[float], beta: list[float]) -> list[float]:
    return [scaled[index] + beta[index] for index in range(len(scaled))]


def calculate_attention_sublayer(
    tokens: list[str],
    input_vectors: list[list[float]],
    attention_outputs: list[list[float]],
    gamma: list[float],
    beta: list[float],
    epsilon: float,
) -> dict[str, Any]:
    residual_vectors: list[list[float]] = []
    normalized_vectors: list[list[float]] = []
    layer_norm_outputs: list[list[float]] = []
    calculations: list[dict[str, Any]] = []

    for position, (input_vector, attention_output) in enumerate(zip(input_vectors, attention_outputs)):
        residual = add_vectors(input_vector, attention_output)
        mean = calculate_mean(residual)
        variance = calculate_population_variance(residual, mean)
        standard_deviation = calculate_standard_deviation(variance, epsilon)
        normalized = normalize_vector(residual, mean, standard_deviation)
        scaled = apply_gamma(normalized, gamma)
        output = apply_beta(scaled, beta)

        displayed_residual = [round(value, 4) for value in residual]
        displayed_normalized = [round(value, 4) for value in normalized]
        displayed_output = [round(value, 4) for value in output]
        residual_vectors.append(displayed_residual)
        normalized_vectors.append(displayed_normalized)
        layer_norm_outputs.append(displayed_output)
        calculations.append(
            {
                "token": tokens[position],
                "position": position,
                "input_vector": input_vector,
                "attention_output": attention_output,
                "residual_terms": [
                    {
                        "dimension": dimension,
                        "input_value": input_vector[dimension],
                        "attention_value": attention_output[dimension],
                        "residual_value": displayed_residual[dimension],
                    }
                    for dimension in range(MODEL_DIMENSION)
                ],
                "residual_vector": displayed_residual,
                "mean": round(mean, 4),
                "variance": round(variance, 4),
                "epsilon": epsilon,
                "standard_deviation": round(standard_deviation, 4),
                "normalization_terms": [
                    {
                        "dimension": dimension,
                        "residual_value": displayed_residual[dimension],
                        "mean": round(mean, 4),
                        "centered_value": round(residual[dimension] - mean, 4),
                        "standard_deviation": round(standard_deviation, 4),
                        "normalized_value": displayed_normalized[dimension],
                        "gamma": gamma[dimension],
                        "beta": beta[dimension],
                        "output_value": displayed_output[dimension],
                    }
                    for dimension in range(MODEL_DIMENSION)
                ],
                "normalized_vector": displayed_normalized,
                "layer_norm_output": displayed_output,
            }
        )

    return {
        "architecture": "post_norm",
        "input_vectors": input_vectors,
        "attention_outputs": attention_outputs,
        "residual_vectors": residual_vectors,
        "gamma": gamma,
        "beta": beta,
        "epsilon": epsilon,
        "normalized_vectors": normalized_vectors,
        "layer_norm_outputs": layer_norm_outputs,
        "calculations": calculations,
    }
