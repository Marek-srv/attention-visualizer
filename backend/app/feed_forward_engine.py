"""Transparent position-wise feed-forward sublayer for Phase 7."""

from __future__ import annotations

from typing import Any

from app.normalization_engine import (
    add_vectors,
    apply_beta,
    apply_gamma,
    calculate_mean,
    calculate_population_variance,
    calculate_standard_deviation,
    normalize_vector,
)


INPUT_DIMENSION = 4
HIDDEN_DIMENSION = 8

DEFAULT_W1 = [
    [0.40, -0.30, 0.20, 0.10, -0.50, 0.30, 0.20, -0.20],
    [-0.20, 0.50, 0.30, -0.40, 0.10, -0.30, 0.40, 0.20],
    [0.30, 0.20, -0.50, 0.40, 0.20, 0.10, -0.20, 0.50],
    [0.10, -0.20, 0.40, 0.30, -0.10, 0.50, -0.30, 0.20],
]
DEFAULT_B1 = [-0.10, 0.05, -0.15, 0.10, 0.00, -0.05, 0.08, -0.12]
DEFAULT_W2 = [
    [0.30, -0.20, 0.10, 0.40],
    [-0.10, 0.40, 0.20, -0.30],
    [0.50, 0.10, -0.20, 0.20],
    [0.20, -0.30, 0.40, 0.10],
    [-0.40, 0.20, 0.30, -0.10],
    [0.10, 0.50, -0.30, 0.20],
    [0.30, 0.20, 0.10, -0.40],
    [-0.20, 0.10, 0.50, 0.30],
]
DEFAULT_B2 = [0.05, -0.04, 0.03, 0.02]
DEFAULT_LN2_GAMMA = [1.0, 1.0, 1.0, 1.0]
DEFAULT_LN2_BETA = [0.0, 0.0, 0.0, 0.0]


def relu(value: float) -> float:
    return max(0.0, value)


def project_with_bias(
    input_vector: list[float], weight_matrix: list[list[float]], bias: list[float]
) -> tuple[list[float], list[dict[str, Any]]]:
    output: list[float] = []
    calculations: list[dict[str, Any]] = []
    for output_dimension in range(len(bias)):
        terms = []
        weighted_sum = 0.0
        for input_dimension, input_value in enumerate(input_vector):
            weight_value = weight_matrix[input_dimension][output_dimension]
            product = input_value * weight_value
            weighted_sum += product
            terms.append(
                {
                    "input_dimension": input_dimension,
                    "input_value": input_value,
                    "weight_value": weight_value,
                    "product": round(product, 4),
                }
            )
        result = weighted_sum + bias[output_dimension]
        output.append(result)
        calculations.append(
            {
                "output_dimension": output_dimension,
                "terms": terms,
                "weighted_sum": round(weighted_sum, 4),
                "bias": bias[output_dimension],
                "result": round(result, 4),
            }
        )
    return output, calculations


def calculate_feed_forward_sublayer(
    tokens: list[str],
    input_vectors: list[list[float]],
    input_weights: list[list[float]],
    input_bias: list[float],
    output_weights: list[list[float]],
    output_bias: list[float],
    gamma: list[float],
    beta: list[float],
    epsilon: float,
) -> dict[str, Any]:
    pre_activation_vectors = []
    activated_vectors = []
    feed_forward_outputs = []
    residual_vectors = []
    normalized_vectors = []
    transformer_block_outputs = []
    calculations = []

    for position, input_vector in enumerate(input_vectors):
        pre_activation, hidden_projection = project_with_bias(input_vector, input_weights, input_bias)
        activated = [relu(value) for value in pre_activation]
        ffn_output, output_projection = project_with_bias(activated, output_weights, output_bias)
        residual = add_vectors(input_vector, ffn_output)
        mean = calculate_mean(residual)
        variance = calculate_population_variance(residual, mean)
        standard_deviation = calculate_standard_deviation(variance, epsilon)
        normalized = normalize_vector(residual, mean, standard_deviation)
        final_output = apply_beta(apply_gamma(normalized, gamma), beta)

        displayed_pre = [round(value, 4) for value in pre_activation]
        displayed_activated = [round(value, 4) for value in activated]
        displayed_ffn = [round(value, 4) for value in ffn_output]
        displayed_residual = [round(value, 4) for value in residual]
        displayed_normalized = [round(value, 4) for value in normalized]
        displayed_final = [round(value, 4) for value in final_output]
        pre_activation_vectors.append(displayed_pre)
        activated_vectors.append(displayed_activated)
        feed_forward_outputs.append(displayed_ffn)
        residual_vectors.append(displayed_residual)
        normalized_vectors.append(displayed_normalized)
        transformer_block_outputs.append(displayed_final)

        hidden_calculations = []
        for neuron, calculation in enumerate(hidden_projection):
            hidden_calculations.append(
                {
                    "hidden_neuron": neuron,
                    "terms": calculation["terms"],
                    "weighted_sum": calculation["weighted_sum"],
                    "bias": calculation["bias"],
                    "pre_activation": displayed_pre[neuron],
                    "activated": displayed_activated[neuron],
                    "is_active": pre_activation[neuron] > 0,
                }
            )

        calculations.append(
            {
                "token": tokens[position],
                "position": position,
                "input_vector": input_vector,
                "hidden_calculations": hidden_calculations,
                "pre_activation_vector": displayed_pre,
                "activated_vector": displayed_activated,
                "output_calculations": [
                    {
                        "output_dimension": dimension,
                        "terms": calculation["terms"],
                        "weighted_sum": calculation["weighted_sum"],
                        "bias": calculation["bias"],
                        "output": displayed_ffn[dimension],
                    }
                    for dimension, calculation in enumerate(output_projection)
                ],
                "feed_forward_output": displayed_ffn,
                "residual_terms": [
                    {
                        "dimension": dimension,
                        "input_value": input_vector[dimension],
                        "feed_forward_value": displayed_ffn[dimension],
                        "residual_value": displayed_residual[dimension],
                    }
                    for dimension in range(INPUT_DIMENSION)
                ],
                "residual_vector": displayed_residual,
                "mean": round(mean, 4),
                "variance": round(variance, 4),
                "standard_deviation": round(standard_deviation, 4),
                "normalized_vector": displayed_normalized,
                "normalization_terms": [
                    {
                        "dimension": dimension,
                        "normalized_value": displayed_normalized[dimension],
                        "gamma": gamma[dimension],
                        "beta": beta[dimension],
                        "output_value": displayed_final[dimension],
                    }
                    for dimension in range(INPUT_DIMENSION)
                ],
                "transformer_block_output": displayed_final,
            }
        )

    return {
        "input_dimension": INPUT_DIMENSION,
        "hidden_dimension": HIDDEN_DIMENSION,
        "activation": "relu",
        "input_vectors": input_vectors,
        "input_weight_matrix": input_weights,
        "input_bias": input_bias,
        "pre_activation_vectors": pre_activation_vectors,
        "activated_vectors": activated_vectors,
        "output_weight_matrix": output_weights,
        "output_bias": output_bias,
        "feed_forward_outputs": feed_forward_outputs,
        "residual_vectors": residual_vectors,
        "normalization": {"gamma": gamma, "beta": beta, "epsilon": epsilon},
        "normalized_vectors": normalized_vectors,
        "transformer_block_outputs": transformer_block_outputs,
        "calculations": calculations,
    }
