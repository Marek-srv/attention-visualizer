import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ToyInspectResponse } from "../../types/api";
import useToyInspection from "./useToyInspection";

const mocks = vi.hoisted(() => ({ inspectToy: vi.fn() }));

vi.mock("../../api/endpoints", () => ({ api: { inspectToy: mocks.inspectToy } }));

describe("useToyInspection request timing", () => {
  beforeEach(() => {
    mocks.inspectToy.mockReset();
    mocks.inspectToy.mockResolvedValue({} as ToyInspectResponse);
  });

  it("runs initially and on an explicit run nonce, not on each prompt edit", async () => {
    const { rerender } = renderHook(
      ({ prompt, runNonce }: { prompt: string; runNonce: number }) => useToyInspection(prompt, runNonce),
      { initialProps: { prompt: "I love", runNonce: 1 } },
    );

    await waitFor(() => expect(mocks.inspectToy).toHaveBeenCalledTimes(1));
    expect(mocks.inspectToy).toHaveBeenLastCalledWith({ text: "I love" }, expect.any(Object));

    rerender({ prompt: "I love music", runNonce: 1 });
    await new Promise((resolve) => window.setTimeout(resolve, 20));
    expect(mocks.inspectToy).toHaveBeenCalledTimes(1);

    rerender({ prompt: "I love music", runNonce: 2 });
    await waitFor(() => expect(mocks.inspectToy).toHaveBeenCalledTimes(2));
    expect(mocks.inspectToy).toHaveBeenLastCalledWith({ text: "I love music" }, expect.any(Object));
  });
});

