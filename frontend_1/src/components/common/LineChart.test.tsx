import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LineChart from "./LineChart";

const animation = vi.hoisted(() => ({
  fromTo: vi.fn(),
  set: vi.fn(),
  killTweensOf: vi.fn(),
  revert: vi.fn(),
}));
const motion = vi.hoisted(() => ({ reduced: false }));

vi.mock("gsap", () => ({
  gsap: {
    context: (run: () => void) => {
      run();
      return { revert: animation.revert };
    },
    fromTo: animation.fromTo,
    set: animation.set,
    killTweensOf: animation.killTweensOf,
  },
}));

vi.mock("../../hooks/useReducedMotion", () => ({
  usePrefersReducedMotion: () => motion.reduced,
}));

const first = { epoch: 1, training: 1.2, validation: 1.4 };
const second = { epoch: 2, training: 0.9, validation: 1.1 };

describe("LineChart metric motion", () => {
  beforeEach(() => {
    animation.fromTo.mockClear();
    animation.set.mockClear();
    animation.killTweensOf.mockClear();
    animation.revert.mockClear();
    motion.reduced = false;
  });

  it("animates only metric points newly added to the backend history", () => {
    const { rerender } = render(<LineChart points={[first]} title="Loss" valueLabel="Loss" />);
    expect(animation.fromTo).toHaveBeenCalledTimes(1);
    expect((animation.fromTo.mock.calls[0]?.[0] as Element).getAttribute("data-chart-epoch")).toBe("1");

    rerender(<LineChart points={[first, second]} title="Loss" valueLabel="Loss" />);
    expect(animation.fromTo).toHaveBeenCalledTimes(2);
    expect((animation.fromTo.mock.calls[1]?.[0] as Element).getAttribute("data-chart-epoch")).toBe("2");

    rerender(<LineChart points={[first, second]} title="Loss" valueLabel="Loss" selectedEpoch={2} />);
    expect(animation.fromTo).toHaveBeenCalledTimes(2);
  });

  it("updates instantly and clears stale animation styles when reduced motion is requested", () => {
    motion.reduced = true;
    render(<LineChart points={[first]} title="Perplexity" valueLabel="Perplexity" />);

    expect(animation.fromTo).not.toHaveBeenCalled();
    expect(animation.killTweensOf).toHaveBeenCalledTimes(1);
    expect(animation.set).toHaveBeenCalledTimes(1);
  });
});
