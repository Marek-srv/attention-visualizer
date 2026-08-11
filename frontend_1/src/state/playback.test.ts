import { describe, expect, it } from "vitest";

import { createInitialExplainerState, explainerReducer } from "./reducer";
import { STAGE_IDS } from "./stageRegistry";

describe("guided playback state", () => {
  it("stops at the final stage and never loops", () => {
    let state = explainerReducer(createInitialExplainerState(), { type: "play" });
    for (let index = 0; index < STAGE_IDS.length + 3; index += 1) {
      state = explainerReducer(state, { type: "next-stage" });
    }

    expect(state.playback.currentStageIndex).toBe(STAGE_IDS.length - 1);
    expect(state.selectedStage).toBe("next-token");
    expect(state.playback.isPlaying).toBe(false);
  });

  it("restart returns to Text without starting an automatic loop", () => {
    let state = explainerReducer(createInitialExplainerState(), { type: "select-stage", stage: "next-token" });
    state = explainerReducer(state, { type: "restart-playback" });
    expect(state.selectedStage).toBe("text");
    expect(state.playback).toMatchObject({ currentStageIndex: 0, isPlaying: false });
  });
});
