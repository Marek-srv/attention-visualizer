import { describe, expect, it } from "vitest";

import { readableApiError } from "./errors";

describe("readableApiError", () => {
  it("uses a backend string detail", () => {
    expect(readableApiError({ detail: "no trained checkpoint is loaded" }, "Fallback")).toBe(
      "no trained checkpoint is loaded",
    );
  });

  it("uses a structured pretrained-service message", () => {
    expect(readableApiError({ detail: { code: "download_failed", message: "Model download failed." } }, "Fallback")).toBe(
      "Model download failed.",
    );
  });

  it("formats Pydantic validation details without object coercion", () => {
    const message = readableApiError({
      detail: [{ type: "greater_than", loc: ["body", "temperature"], msg: "Input should be greater than 0" }],
    }, "Fallback");
    expect(message).toBe("temperature: Input should be greater than 0");
    expect(message).not.toContain("[object Object]");
  });
});
