import json
import math

import pytest
from fastapi.testclient import TestClient

from app.attention_engine import stable_softmax
from app.main import app
from app.projection_engine import DEFAULT_WK, DEFAULT_WQ, DEFAULT_WV
from app.multi_head_engine import DEFAULT_WO
from app.normalization_engine import calculate_attention_sublayer
from app.feed_forward_engine import (
    DEFAULT_B1,
    DEFAULT_B2,
    DEFAULT_LN2_BETA,
    DEFAULT_LN2_GAMMA,
    DEFAULT_W1,
    DEFAULT_W2,
    calculate_feed_forward_sublayer,
)


client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_inspect_text() -> None:
    response = client.post("/api/inspect", json={"text": "I love"})

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "I love"
    assert body["character_count"] == 6
    assert body["phase"] == 7
    assert body["token_count"] == 2
    assert body["vocabulary_size"] == 10
    assert body["embedding_dimension"] == 4
    assert [(token["token"], token["token_id"], token["position"]) for token in body["tokens"]] == [
        ("I", 1, 0),
        ("love", 2, 1),
    ]
    assert body["tokens"][0]["position_embedding"] == [0.0, 1.0, 0.0, 1.0]
    assert body["tokens"][0]["combined_embedding"] == [0.12, 0.69, 0.48, 1.07]


def test_unknown_token_uses_unk_embedding() -> None:
    response = client.post("/api/inspect", json={"text": "Hello, AI!"})

    assert response.status_code == 200
    body = response.json()
    assert body["token_count"] == 4
    assert [token["token"] for token in body["tokens"]] == ["Hello", ",", "AI", "!"]
    assert [token["normalized"] for token in body["tokens"]] == ["<UNK>", "<UNK>", "AI", "!"]
    assert body["tokens"][0]["token_id"] == 0
    assert body["tokens"][0]["token_embedding"] == [0.0, 0.0, 0.0, 0.0]


def test_rejects_empty_text() -> None:
    response = client.post("/api/inspect", json={"text": ""})

    assert response.status_code == 422


def test_default_matrices_return_four_dimensional_qkv() -> None:
    response = client.post("/api/inspect", json={"text": "I love"})

    assert response.status_code == 200
    body = response.json()
    assert body["weights"] == {
        "query": DEFAULT_WQ,
        "key": DEFAULT_WK,
        "value": DEFAULT_WV,
        "output": DEFAULT_WO,
    }
    assert len(body["projections"]) == 2
    for projection in body["projections"]:
        assert len(projection["query"]) == 4
        assert len(projection["key"]) == 4
        assert len(projection["value"]) == 4
        assert len(projection["breakdown"]["query"]) == 4
        assert len(projection["breakdown"]["key"]) == 4
        assert len(projection["breakdown"]["value"]) == 4


def test_identity_query_weights_return_combined_input() -> None:
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    response = client.post(
        "/api/inspect",
        json={
            "text": "I love",
            "weights": {"query": identity, "key": DEFAULT_WK, "value": DEFAULT_WV},
        },
    )

    assert response.status_code == 200
    body = response.json()
    for token, projection in zip(body["tokens"], body["projections"]):
        assert projection["query"] == [round(value, 4) for value in token["combined_embedding"]]
        assert projection["input_vector"] == token["combined_embedding"]


def test_custom_matrices_change_projection_results() -> None:
    default_body = client.post("/api/inspect", json={"text": "I love"}).json()
    custom = [[0.25, 0.25, 0.25, 0.25] for _ in range(4)]
    custom_body = client.post(
        "/api/inspect",
        json={
            "text": "I love",
            "weights": {"query": custom, "key": custom, "value": custom},
        },
    ).json()

    assert custom_body["projections"][0]["query"] != default_body["projections"][0]["query"]
    assert custom_body["projections"][0]["query"] == custom_body["projections"][0]["key"]
    assert custom_body["projections"][0]["key"] == custom_body["projections"][0]["value"]


def test_invalid_three_by_three_matrix_returns_422() -> None:
    invalid = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    response = client.post(
        "/api/inspect",
        json={
            "text": "I love",
            "weights": {"query": invalid, "key": DEFAULT_WK, "value": DEFAULT_WV},
        },
    )

    assert response.status_code == 422
    assert "exactly 4 rows" in response.text


def test_raw_and_scaled_attention_scores() -> None:
    body = client.post("/api/inspect", json={"text": "I love"}).json()
    projections = body["projections"]
    attention = body["attention"]

    for query_position, query in enumerate(projections):
        for key_position, key in enumerate(projections):
            expected_raw = round(sum(q * k for q, k in zip(query["query"], key["key"])), 4)
            assert attention["raw_scores"][query_position][key_position] == expected_raw
            assert attention["scaled_scores"][query_position][key_position] == round(expected_raw / math.sqrt(4), 4)


def test_causal_mask_and_attention_row_sums() -> None:
    attention = client.post("/api/inspect", json={"text": "I love music"}).json()["attention"]

    assert attention["causal_mask"] == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    assert attention["masked_scores"][0][1:] == [None, None]
    assert attention["attention_weights"][0] == [1.0, 0.0, 0.0]
    assert attention["attention_weights"][1][2] == 0.0
    for row_sum in attention["row_sums"]:
        assert row_sum == pytest.approx(1.0, abs=0.0001)


def test_stable_softmax_handles_large_values() -> None:
    weights = stable_softmax([10000.0, 10001.0, 9999.0])

    assert all(math.isfinite(weight) for weight in weights)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[1] > weights[0] > weights[2]


def test_context_vectors_equal_attention_weights_times_values() -> None:
    body = client.post("/api/inspect", json={"text": "I love music"}).json()
    attention = body["attention"]
    values = [projection["value"] for projection in body["projections"]]

    assert len(attention["context_vectors"]) == 3
    for query_position, context in enumerate(attention["context_vectors"]):
        assert len(context) == 4
        for dimension, actual in enumerate(context):
            expected = round(
                sum(attention["attention_weights"][query_position][key] * values[key][dimension] for key in range(3)),
                4,
            )
            assert actual == expected


def test_empty_and_non_finite_weight_matrices_return_422() -> None:
    empty_response = client.post(
        "/api/inspect",
        json={"text": "I love", "weights": {"query": [], "key": DEFAULT_WK, "value": DEFAULT_WV}},
    )
    infinite_response = client.post(
        "/api/inspect",
        content=json.dumps({
            "text": "I love",
            "weights": {
                "query": [[float("inf"), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "key": DEFAULT_WK,
                "value": DEFAULT_WV,
            },
        }),
        headers={"Content-Type": "application/json"},
    )

    assert empty_response.status_code == 422
    assert infinite_response.status_code == 422


def test_multi_head_configuration_and_dimension_splits() -> None:
    body = client.post("/api/inspect", json={"text": "I love"}).json()
    multi_head = body["multi_head_attention"]

    assert multi_head["number_of_heads"] == 2
    assert multi_head["head_dimension"] == 2
    assert multi_head["scale_factor"] == pytest.approx(math.sqrt(2), abs=0.0001)
    assert multi_head["heads"][0]["dimension_indices"] == [0, 1]
    assert multi_head["heads"][1]["dimension_indices"] == [2, 3]
    for token_position, projection in enumerate(body["projections"]):
        assert multi_head["heads"][0]["query_vectors"][token_position] == projection["query"][:2]
        assert multi_head["heads"][1]["query_vectors"][token_position] == projection["query"][2:]


def test_each_head_masks_scales_and_returns_contexts() -> None:
    multi_head = client.post("/api/inspect", json={"text": "I love music"}).json()["multi_head_attention"]

    for head in multi_head["heads"]:
        assert head["scaled_scores"][1][0] == round(head["raw_scores"][1][0] / math.sqrt(2), 4)
        assert head["masked_scores"][0][1:] == [None, None]
        assert head["attention_weights"][0] == [1.0, 0.0, 0.0]
        for row_sum in head["row_sums"]:
            assert row_sum == pytest.approx(1.0, abs=0.0001)
        assert all(len(context) == 2 for context in head["context_vectors"])
    assert all(len(context) == 4 for context in multi_head["concatenated_contexts"])


def test_output_projection_equals_concatenated_context_times_wo() -> None:
    multi_head = client.post("/api/inspect", json={"text": "I love"}).json()["multi_head_attention"]

    for context, output in zip(multi_head["concatenated_contexts"], multi_head["projected_outputs"]):
        expected = [
            round(sum(context[input_dimension] * DEFAULT_WO[input_dimension][output_dimension] for input_dimension in range(4)), 4)
            for output_dimension in range(4)
        ]
        assert output == expected


def test_identity_and_custom_output_weights() -> None:
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    custom = [[0.2, 0.2, 0.2, 0.2] for _ in range(4)]
    base_weights = {"query": DEFAULT_WQ, "key": DEFAULT_WK, "value": DEFAULT_WV}
    identity_body = client.post(
        "/api/inspect", json={"text": "I love", "weights": {**base_weights, "output": identity}}
    ).json()
    custom_body = client.post(
        "/api/inspect", json={"text": "I love", "weights": {**base_weights, "output": custom}}
    ).json()

    identity_multi_head = identity_body["multi_head_attention"]
    assert identity_multi_head["projected_outputs"] == identity_multi_head["concatenated_contexts"]
    assert custom_body["multi_head_attention"]["projected_outputs"] != identity_multi_head["projected_outputs"]


def test_invalid_output_matrix_returns_422() -> None:
    response = client.post(
        "/api/inspect",
        json={
            "text": "I love",
            "weights": {
                "query": DEFAULT_WQ,
                "key": DEFAULT_WK,
                "value": DEFAULT_WV,
                "output": [[1, 0], [0, 1]],
            },
        },
    )

    assert response.status_code == 422


def test_multi_head_response_contains_only_finite_numbers() -> None:
    multi_head = client.post("/api/inspect", json={"text": "I love AI!"}).json()["multi_head_attention"]

    def assert_finite(value: object) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, list):
            for item in value:
                assert_finite(item)
        elif isinstance(value, dict):
            for item in value.values():
                assert_finite(item)

    assert_finite(multi_head)


def test_residual_mean_variance_and_standard_deviation() -> None:
    body = client.post("/api/inspect", json={"text": "I love"}).json()
    sublayer = body["attention_sublayer"]

    for token, attention_output, calculation in zip(
        body["tokens"], body["multi_head_attention"]["projected_outputs"], sublayer["calculations"]
    ):
        residual = [input_value + attention_value for input_value, attention_value in zip(token["combined_embedding"], attention_output)]
        expected_mean = sum(residual) / 4
        expected_variance = sum((value - expected_mean) ** 2 for value in residual) / 4
        expected_standard_deviation = math.sqrt(expected_variance + sublayer["epsilon"])
        assert calculation["residual_vector"] == [round(value, 4) for value in residual]
        assert calculation["mean"] == round(expected_mean, 4)
        assert calculation["variance"] == round(expected_variance, 4)
        assert calculation["standard_deviation"] == round(expected_standard_deviation, 4)


def test_default_layer_normalization_statistics_and_affine_identity() -> None:
    sublayer = client.post("/api/inspect", json={"text": "I love"}).json()["attention_sublayer"]

    assert sublayer["architecture"] == "post_norm"
    assert sublayer["gamma"] == [1.0, 1.0, 1.0, 1.0]
    assert sublayer["beta"] == [0.0, 0.0, 0.0, 0.0]
    for normalized, output in zip(sublayer["normalized_vectors"], sublayer["layer_norm_outputs"]):
        mean = sum(normalized) / 4
        variance = sum((value - mean) ** 2 for value in normalized) / 4
        assert mean == pytest.approx(0.0, abs=0.0001)
        assert variance == pytest.approx(1.0, abs=0.001)
        assert output == normalized


def test_custom_gamma_and_beta_scale_and_shift_each_dimension() -> None:
    gamma = [2.0, 0.5, -1.0, 1.5]
    beta = [0.1, -0.2, 0.3, 0.4]
    sublayer = client.post(
        "/api/inspect",
        json={"text": "I love", "normalization": {"gamma": gamma, "beta": beta, "epsilon": 0.00001}},
    ).json()["attention_sublayer"]

    for calculation, output in zip(sublayer["calculations"], sublayer["layer_norm_outputs"]):
        residual = [
            term["input_value"] + term["attention_value"] for term in calculation["residual_terms"]
        ]
        mean = sum(residual) / 4
        variance = sum((value - mean) ** 2 for value in residual) / 4
        normalized = [(value - mean) / math.sqrt(variance + sublayer["epsilon"]) for value in residual]
        assert output == [round(gamma[index] * normalized[index] + beta[index], 4) for index in range(4)]


def test_constant_vectors_remain_finite() -> None:
    result = calculate_attention_sublayer(
        ["constant"], [[2.0, 2.0, 2.0, 2.0]], [[0.0, 0.0, 0.0, 0.0]], [1.0] * 4, [0.0] * 4, 0.00001
    )

    assert result["normalized_vectors"] == [[0.0, 0.0, 0.0, 0.0]]
    assert result["layer_norm_outputs"] == [[0.0, 0.0, 0.0, 0.0]]
    assert all(math.isfinite(value) for value in result["layer_norm_outputs"][0])


@pytest.mark.parametrize(
    "normalization",
    [
        {"gamma": [1, 1, 1], "beta": [0, 0, 0, 0], "epsilon": 0.00001},
        {"gamma": [1, 1, 1, 1], "beta": [0, 0, 0], "epsilon": 0.00001},
        {"gamma": [1, 1, 1, 1], "beta": [0, 0, 0, 0], "epsilon": 0},
        {"gamma": [1, 1, 1, 1], "beta": [0, 0, 0, 0], "epsilon": -0.1},
    ],
)
def test_invalid_normalization_returns_422(normalization: dict[str, object]) -> None:
    response = client.post("/api/inspect", json={"text": "I love", "normalization": normalization})

    assert response.status_code == 422


def test_non_finite_normalization_values_return_json_safe_422() -> None:
    payload = {
        "text": "I love",
        "normalization": {"gamma": [float("nan"), 1, 1, 1], "beta": [0, 0, 0, 0], "epsilon": float("inf")},
    }
    response = client.post(
        "/api/inspect", content=json.dumps(payload), headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    assert "finite" in response.text


def test_feed_forward_parameter_shapes_and_vector_dimensions() -> None:
    sublayer = client.post("/api/inspect", json={"text": "I love"}).json()["feed_forward_sublayer"]

    assert len(sublayer["input_weight_matrix"]) == 4
    assert all(len(row) == 8 for row in sublayer["input_weight_matrix"])
    assert len(sublayer["input_bias"]) == 8
    assert len(sublayer["output_weight_matrix"]) == 8
    assert all(len(row) == 4 for row in sublayer["output_weight_matrix"])
    assert len(sublayer["output_bias"]) == 4
    assert all(len(vector) == 8 for vector in sublayer["activated_vectors"])
    assert all(len(vector) == 4 for vector in sublayer["feed_forward_outputs"])


def test_hidden_projection_and_relu_are_correct() -> None:
    sublayer = client.post("/api/inspect", json={"text": "I love"}).json()["feed_forward_sublayer"]

    saw_active = False
    saw_inactive = False
    for calculation in sublayer["calculations"]:
        input_vector = calculation["input_vector"]
        for hidden in calculation["hidden_calculations"]:
            neuron = hidden["hidden_neuron"]
            expected = round(
                sum(input_vector[dimension] * DEFAULT_W1[dimension][neuron] for dimension in range(4))
                + DEFAULT_B1[neuron],
                4,
            )
            assert hidden["pre_activation"] == expected
            assert hidden["activated"] == max(0.0, expected)
            saw_active |= hidden["is_active"]
            saw_inactive |= not hidden["is_active"]
    assert saw_active and saw_inactive


def test_ffn_output_and_second_residual_are_correct() -> None:
    sublayer = client.post("/api/inspect", json={"text": "I love"}).json()["feed_forward_sublayer"]

    for calculation in sublayer["calculations"]:
        activated = calculation["activated_vector"]
        expected_ffn = [
            round(sum(activated[hidden] * DEFAULT_W2[hidden][dimension] for hidden in range(8)) + DEFAULT_B2[dimension], 4)
            for dimension in range(4)
        ]
        assert calculation["feed_forward_output"] == pytest.approx(expected_ffn, abs=0.0001)
        assert calculation["residual_vector"] == pytest.approx(
            [round(calculation["input_vector"][dimension] + calculation["feed_forward_output"][dimension], 4) for dimension in range(4)],
            abs=0.0001,
        )


def test_layer_norm2_population_variance_and_final_statistics() -> None:
    sublayer = client.post("/api/inspect", json={"text": "I love"}).json()["feed_forward_sublayer"]

    for calculation, output in zip(sublayer["calculations"], sublayer["transformer_block_outputs"]):
        residual = calculation["residual_vector"]
        mean = sum(residual) / 4
        population_variance = sum((value - mean) ** 2 for value in residual) / 4
        assert calculation["variance"] == pytest.approx(population_variance, abs=0.0001)
        assert sum(output) / 4 == pytest.approx(0.0, abs=0.0001)
        output_mean = sum(output) / 4
        assert sum((value - output_mean) ** 2 for value in output) / 4 == pytest.approx(1.0, abs=0.001)


def test_custom_w1_and_w2_change_feed_forward_results() -> None:
    default = client.post("/api/inspect", json={"text": "I love"}).json()["feed_forward_sublayer"]
    custom_w1 = [[value * 0.5 for value in row] for row in DEFAULT_W1]
    custom_w2 = [[value * -0.5 for value in row] for row in DEFAULT_W2]
    changed_w1 = client.post(
        "/api/inspect", json={"text": "I love", "feed_forward": {"input_weights": custom_w1}}
    ).json()["feed_forward_sublayer"]
    changed_w2 = client.post(
        "/api/inspect", json={"text": "I love", "feed_forward": {"output_weights": custom_w2}}
    ).json()["feed_forward_sublayer"]

    assert changed_w1["activated_vectors"] != default["activated_vectors"]
    assert changed_w2["feed_forward_outputs"] != default["feed_forward_outputs"]


@pytest.mark.parametrize(
    "feed_forward",
    [
        {"input_weights": [[0] * 8 for _ in range(3)]},
        {"input_bias": [0] * 7},
        {"output_weights": [[0] * 4 for _ in range(7)]},
        {"output_bias": [0] * 3},
    ],
)
def test_invalid_feed_forward_shapes_return_422(feed_forward: dict[str, object]) -> None:
    assert client.post("/api/inspect", json={"text": "I love", "feed_forward": feed_forward}).status_code == 422


def test_constant_feed_forward_input_remains_finite() -> None:
    result = calculate_feed_forward_sublayer(
        ["constant"], [[1.0, 1.0, 1.0, 1.0]], DEFAULT_W1, DEFAULT_B1, DEFAULT_W2, DEFAULT_B2,
        DEFAULT_LN2_GAMMA, DEFAULT_LN2_BETA, 0.00001,
    )

    def values(value: object):
        if isinstance(value, float):
            yield value
        elif isinstance(value, list):
            for item in value:
                yield from values(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from values(item)

    assert all(math.isfinite(value) for value in values(result))
