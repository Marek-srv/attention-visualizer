import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WorkspaceContext } from "../../app/workspaceContext";
import type { PretrainedStatus } from "../../types/api";
import RealModelPage from "./RealModelPage";

const apiMocks = vi.hoisted(() => ({
  pretrainedStatus: vi.fn(),
  loadPretrained: vi.fn(),
  inspectPretrained: vi.fn(),
  predictPretrained: vi.fn(),
}));

vi.mock("../../api/endpoints", () => ({ api: apiMocks }));

function status(state: PretrainedStatus["status"], error: string | null = null): PretrainedStatus {
  return {
    status: state,
    loaded: state === "loaded",
    loading: state === "loading",
    model_name: "sshleifer/tiny-gpt2",
    device: "cpu",
    dependencies_available: { torch: true, transformers: true },
    model: state === "loaded" ? {
      name: "sshleifer/tiny-gpt2",
      device: "cpu",
      number_of_layers: 2,
      number_of_heads: 2,
      hidden_dimension: 2,
      vocabulary_size: 50257,
      context_length: 1024,
      attention_implementation: "eager",
    } : null,
    error,
  };
}

function renderPage() {
  return render(
    <WorkspaceContext.Provider value={{
      mode: "real-model",
      prompt: "I love",
      runNonce: 1,
      learningMode: "explore",
      textbookOpen: true,
      setPrompt: vi.fn(),
      requestRun: vi.fn(),
    }}>
      <RealModelPage />
    </WorkspaceContext.Provider>,
  );
}

describe("RealModelPage loading states", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.loadPretrained.mockResolvedValue(status("loaded"));
  });

  afterEach(() => vi.useRealTimers());

  it("keeps the pretrained model unloaded until Load Model is explicitly selected", async () => {
    apiMocks.pretrainedStatus.mockResolvedValue(status("not_loaded"));
    renderPage();

    expect(await screen.findByRole("button", { name: "Load pretrained model" })).toBeEnabled();
    expect(apiMocks.loadPretrained).not.toHaveBeenCalled();
  });

  it("renders the backend loading state without offering a second load", async () => {
    apiMocks.pretrainedStatus.mockResolvedValue(status("loading"));
    renderPage();

    expect(await screen.findByRole("button", { name: "Loading model…" })).toBeDisabled();
    expect(apiMocks.loadPretrained).not.toHaveBeenCalled();
  });

  it("shows the backend failure and provides an explicit retry", async () => {
    const user = userEvent.setup();
    apiMocks.pretrainedStatus.mockResolvedValue(status("failed", "Network unavailable"));
    renderPage();

    expect(await screen.findByText("Network unavailable")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry loading model" }));
    expect(apiMocks.loadPretrained).toHaveBeenCalledTimes(1);
  });

  it("polls only a backend loading state until an in-flight external load resolves", async () => {
    vi.useFakeTimers();
    apiMocks.pretrainedStatus
      .mockResolvedValueOnce(status("loading"))
      .mockResolvedValueOnce(status("loaded"));
    renderPage();

    await act(async () => undefined);
    expect(screen.getByRole("button", { name: "Loading model…" })).toBeDisabled();

    await act(async () => vi.advanceTimersByTimeAsync(1200));
    expect(screen.getByText("Model loaded. Request only the layer and head you want to inspect.")).toBeInTheDocument();
    expect(apiMocks.pretrainedStatus).toHaveBeenCalledTimes(2);

    await act(async () => vi.advanceTimersByTimeAsync(5000));
    expect(apiMocks.pretrainedStatus).toHaveBeenCalledTimes(2);
  });
});
