import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { motionDuration, usePrefersReducedMotion } from "./useReducedMotion";
import { installMatchMedia } from "../test/factories";

describe("reduced motion", () => {
  it("tracks the operating-system preference", () => {
    const media = installMatchMedia(true);
    const { result } = renderHook(() => usePrefersReducedMotion());
    expect(result.current).toBe(true);

    act(() => media.setMatches(false));
    expect(result.current).toBe(false);
  });

  it("reduces controlled animation durations to zero", () => {
    expect(motionDuration(420, true)).toBe(0);
    expect(motionDuration(420, false)).toBe(420);
  });
});
