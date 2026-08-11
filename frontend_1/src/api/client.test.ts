import { describe, expect, it, vi } from "vitest";

import { ApiClient } from "./client";
import type { ApiError } from "./errors";
import { createAttentionApi } from "./endpoints";
import { createHealthResponse, jsonResponse } from "../test/factories";

describe("ApiClient", () => {
  it("uses the configured base URL and forwards an AbortSignal", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(createHealthResponse()));
    const client = new ApiClient({ baseUrl: "https://example.test/", fetchImplementation: fetchMock });
    const controller = client.createController();

    await expect(client.get("/api/health", { signal: controller.signal })).resolves.toEqual(createHealthResponse());
    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.test/api/health",
      expect.objectContaining({ method: "GET", signal: controller.signal }),
    );
  });

  it("turns a non-2xx response into a readable ApiError", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      jsonResponse({ detail: { code: "model_not_loaded", message: "Load a checkpoint first." } }, 409, "Conflict"),
    );
    const client = new ApiClient({ fetchImplementation: fetchMock });

    await expect(client.get("/api/model/status")).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      message: "Load a checkpoint first.",
    } satisfies Partial<ApiError>);
  });

  it("reports transport failures as a disconnected backend", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Network failed"));
    const client = new ApiClient({ fetchImplementation: fetchMock });

    await expect(client.get("/api/health")).rejects.toThrow("Backend disconnected");
  });
});

describe("typed endpoints", () => {
  it("posts toy inspection to the repository's existing route", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ text: "I love" }));
    const api = createAttentionApi(new ApiClient({ fetchImplementation: fetchMock }));
    const controller = new AbortController();

    await api.inspectToy({ text: "I love" }, { signal: controller.signal });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/inspect",
      expect.objectContaining({
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({ text: "I love" }),
      }),
    );
  });
});
