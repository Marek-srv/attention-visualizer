import { describe, expect, it } from "vitest";

import { initialToyExplainerState, TOY_STAGE_IDS, toyExplainerReducer } from "./toyState";

describe("toy explainer navigation", () => {
  it("keeps a position-based token selection stable between stages", () => {
    const selected = toyExplainerReducer(initialToyExplainerState, { type: "select-token", position: 1 });
    const moved = toyExplainerReducer(selected, { type: "select-stage", stage: "attention" });

    expect(moved.selectedTokenPosition).toBe(1);
    expect(moved.stage).toBe("attention");
  });

  it("stops guided playback at the final stage instead of looping", () => {
    let state = toyExplainerReducer(initialToyExplainerState, { type: "play" });
    for (let index = 0; index < TOY_STAGE_IDS.length + 3; index += 1) {
      state = toyExplainerReducer(state, { type: "next-stage" });
    }

    expect(state.stage).toBe("next-token");
    expect(state.isPlaying).toBe(false);
  });

  it("supports the selectable Attention × V context substage", () => {
    const state = toyExplainerReducer(initialToyExplainerState, { type: "select-attention-view", view: "context_vectors" });
    expect(state.attentionView).toBe("context_vectors");
  });
});
