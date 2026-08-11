import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { ToyInspectResponse } from "../../types/api";
import { initialToyExplainerState } from "../../features/toy-lab/toyState";
import MobileStageFlow from "./MobileStageFlow";

function toyResponse(): ToyInspectResponse {
  const x0 = [0.1, 0.2, 0.3, 0.4];
  const x1 = [0.5, 0.6, 0.7, 0.8];
  const matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const tokens = [
    { token: "I", normalized: "I", token_id: 1, position: 0, token_embedding: x0, position_embedding: x0, combined_embedding: x0 },
    { token: "love", normalized: "love", token_id: 2, position: 1, token_embedding: x1, position_embedding: x1, combined_embedding: x1 },
  ];
  return {
    text: "I love",
    character_count: 6,
    phase: 7,
    token_count: 2,
    vocabulary_size: 10,
    embedding_dimension: 4,
    tokens,
    weights: { query: matrix, key: matrix, value: matrix, output: matrix },
    projections: tokens.map((token) => ({ token: token.token, position: token.position, input_vector: token.combined_embedding, query: token.combined_embedding, key: token.combined_embedding, value: token.combined_embedding, breakdown: { query: [], key: [], value: [] } })),
    attention: { key_dimension: 4, scale_factor: 2, causal_mask: [[true, false], [true, true]], raw_scores: [[1, 0], [0.4, 0.6]], scaled_scores: [[1, 0], [0.4, 0.6]], masked_scores: [[1, null], [0.4, 0.6]], attention_weights: [[1, 0], [0.4, 0.6]], context_vectors: [x0, x1], row_sums: [1, 1], calculations: [], context_calculations: [] },
    multi_head_attention: { model_dimension: 4, number_of_heads: 2, head_dimension: 2, scale_factor: Math.sqrt(2), heads: [], concatenated_contexts: [x0, x1], output_weight_matrix: matrix, projected_outputs: [x0, x1], output_calculations: [] },
    attention_sublayer: { architecture: "post_norm", input_vectors: [x0, x1], attention_outputs: [x0, x1], residual_vectors: [x0, x1], gamma: [1, 1, 1, 1], beta: [0, 0, 0, 0], epsilon: 0.00001, normalized_vectors: [x0, x1], layer_norm_outputs: [x0, x1], calculations: [] },
    feed_forward_sublayer: { input_dimension: 4, hidden_dimension: 8, activation: "relu", input_vectors: [x0, x1], input_weight_matrix: Array.from({ length: 4 }, () => Array(8).fill(0)), input_bias: Array(8).fill(0), pre_activation_vectors: [Array(8).fill(0), Array(8).fill(0)], activated_vectors: [Array(8).fill(0), Array(8).fill(0)], output_weight_matrix: Array.from({ length: 8 }, () => Array(4).fill(0)), output_bias: Array(4).fill(0), feed_forward_outputs: [x0, x1], residual_vectors: [x0, x1], normalization: { gamma: [1, 1, 1, 1], beta: [0, 0, 0, 0], epsilon: 0.00001 }, normalized_vectors: [x0, x1], transformer_block_outputs: [x0, x1], calculations: [] },
  };
}

describe("MobileStageFlow", () => {
  it("keeps every stage navigable in the vertical mobile flow", async () => {
    const user = userEvent.setup();
    const dispatch = vi.fn();
    render(<MobileStageFlow result={toyResponse()} state={initialToyExplainerState} dispatch={dispatch} />);

    expect(screen.getByLabelText("Vertical Transformer stage flow")).toBeInTheDocument();
    expect(screen.getAllByRole("button")).toHaveLength(11);
    await user.click(screen.getByRole("button", { name: /Scaled causal self-attention/i }));
    expect(dispatch).toHaveBeenCalledWith({ type: "select-stage", stage: "attention" });
  });
});

