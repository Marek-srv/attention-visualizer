import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { playbackDelay, useGuidedPlayback } from "./useGuidedPlayback";

afterEach(() => vi.useRealTimers());

describe("useGuidedPlayback", () => {
  it("advances once at the selected speed", () => {
    vi.useFakeTimers();
    const onAdvance = vi.fn();
    const onStop = vi.fn();
    renderHook(() => useGuidedPlayback({
      isPlaying: true,
      currentStageIndex: 1,
      stageCount: 4,
      speed: 1.5,
      baseDelay: 1500,
      onAdvance,
      onStop,
    }));

    act(() => vi.advanceTimersByTime(playbackDelay(1500, 1.5)));
    expect(onAdvance).toHaveBeenCalledOnce();
    expect(onStop).not.toHaveBeenCalled();
  });

  it("stops instead of scheduling another advance at the final stage", () => {
    const onAdvance = vi.fn();
    const onStop = vi.fn();
    renderHook(() => useGuidedPlayback({
      isPlaying: true,
      currentStageIndex: 3,
      stageCount: 4,
      speed: 1,
      onAdvance,
      onStop,
    }));

    expect(onStop).toHaveBeenCalledOnce();
    expect(onAdvance).not.toHaveBeenCalled();
  });
});
