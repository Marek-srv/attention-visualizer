"""Transparent Q, K and V projections for Phase 3."""

from __future__ import annotations

from typing import TypedDict


Matrix = list[list[float]]

DEFAULT_WQ: Matrix = [
    [0.50, -0.20, 0.10, 0.40],
    [0.30, 0.60, -0.40, 0.20],
    [-0.10, 0.30, 0.70, -0.20],
    [0.40, -0.50, 0.20, 0.60],
]

DEFAULT_WK: Matrix = [
    [0.20, 0.50, -0.30, 0.10],
    [-0.40, 0.20, 0.60, 0.30],
    [0.70, -0.10, 0.20, -0.50],
    [0.10, 0.40, 0.30, 0.80],
]

DEFAULT_WV: Matrix = [
    [0.60, 0.10, 0.30, -0.20],
    [0.20, 0.70, -0.10, 0.40],
    [-0.30, 0.20, 0.50, 0.60],
    [0.50, -0.40, 0.20, 0.30],
]


class CalculationTerm(TypedDict):
    input_dimension: int
    input_value: float
    weight_value: float
    product: float


class DimensionBreakdown(TypedDict):
    output_dimension: int
    terms: list[CalculationTerm]
    result: float


def multiply_row_vector(
    input_vector: list[float], weight_matrix: Matrix
) -> tuple[list[float], list[DimensionBreakdown]]:
    """Multiply one row vector by a matrix and expose every scalar operation."""
    output: list[float] = []
    breakdown: list[DimensionBreakdown] = []

    for output_dimension in range(4):
        terms: list[CalculationTerm] = []
        total = 0.0
        for input_dimension in range(4):
            input_value = input_vector[input_dimension]
            weight_value = weight_matrix[input_dimension][output_dimension]
            product = input_value * weight_value
            total += product
            terms.append(
                {
                    "input_dimension": input_dimension,
                    "input_value": input_value,
                    "weight_value": weight_value,
                    "product": round(product, 4),
                }
            )

        result = round(total, 4)
        output.append(result)
        breakdown.append(
            {
                "output_dimension": output_dimension,
                "terms": terms,
                "result": result,
            }
        )

    return output, breakdown


def project_vector(input_vector: list[float], weights: dict[str, Matrix]) -> dict[str, object]:
    """Calculate Q, K and V vectors and their arithmetic breakdowns."""
    query, query_breakdown = multiply_row_vector(input_vector, weights["query"])
    key, key_breakdown = multiply_row_vector(input_vector, weights["key"])
    value, value_breakdown = multiply_row_vector(input_vector, weights["value"])
    return {
        "query": query,
        "key": key,
        "value": value,
        "breakdown": {
            "query": query_breakdown,
            "key": key_breakdown,
            "value": value_breakdown,
        },
    }
