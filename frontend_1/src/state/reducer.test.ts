import { describe, expect, it } from "vitest";

import { createInitialExplainerState, explainerReducer } from "./reducer";

describe("explainerReducer", () => {
  it("navigates stages while preserving the selected token", () => {
    const selected = explainerReducer(createInitialExplainerState(), { type: "select-token", tokenIndex: 1 });
    const qkv = explainerReducer(selected, { type: "select-stage", stage: "qkv" });
    const attention = explainerReducer(qkv, { type: "select-stage", stage: "causal-attention" });

    expect(attention.selectedStage).toBe("causal-attention");
    expect(attention.selectedTokenIndex).toBe(1);
  });

  it("toggles an expanded stage and closes it explicitly", () => {
    const state = createInitialExplainerState();
    const expanded = explainerReducer(state, { type: "expand-stage", stage: "embeddings" });
    expect(expanded.expandedStage).toBe("embeddings");
    expect(explainerReducer(expanded, { type: "expand-stage", stage: "embeddings" }).expandedStage).toBeNull();
    expect(explainerReducer(expanded, { type: "close-stage" }).expandedStage).toBeNull();
  });
});
