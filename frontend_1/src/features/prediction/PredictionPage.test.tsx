import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { WorkspaceContext } from "../../app/workspaceContext";
import { createModelStatus, installMatchMedia } from "../../test/factories";
import type { GenerationResponse } from "../../types/api";
import PredictionPage from "./PredictionPage";

const apiMocks = vi.hoisted(() => ({
  modelStatus: vi.fn(),
  loadModel: vi.fn(),
  predict: vi.fn(),
  generate: vi.fn(),
}));

vi.mock("../../api/endpoints", () => ({ api: apiMocks }));

const generation: GenerationResponse = {
  input_text: "I love",
  input_tokens: ["<BOS>", "I", "love"],
  input_token_ids: [2, 4, 5],
  truncated: false,
  generated_text: "I love you music !",
  generated_tokens: ["you", "music", "!"],
  generated_token_ids: [6, 7, 8],
  strategy: "greedy",
  seed: 42,
  temperature: 1,
  top_k: 2,
  max_new_tokens: 3,
  stop_reason: "max_new_tokens",
  probability_label: "model probability",
  steps: [
    { step: 1, chosen_token: "you", chosen_token_id: 6, chosen_probability: 0.6, top_predictions: [{ token: "you", token_id: 6, logit: 2, probability: 0.6 }], is_eos: false },
    { step: 2, chosen_token: "music", chosen_token_id: 7, chosen_probability: 0.5, top_predictions: [{ token: "music", token_id: 7, logit: 1.7, probability: 0.5 }], is_eos: false },
    { step: 3, chosen_token: "!", chosen_token_id: 8, chosen_probability: 0.4, top_predictions: [{ token: "!", token_id: 8, logit: 1.2, probability: 0.4 }], is_eos: false },
  ],
};

function renderPage() {
  return render(
    <WorkspaceContext.Provider value={{
      mode: "prediction",
      prompt: "I love",
      runNonce: 1,
      learningMode: "explore",
      textbookOpen: true,
      setPrompt: vi.fn(),
      requestRun: vi.fn(),
    }}>
      <PredictionPage />
    </WorkspaceContext.Provider>,
  );
}

describe("PredictionPage generation motion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.modelStatus.mockResolvedValue(createModelStatus({ loaded: true, checkpoint_available: true }));
    apiMocks.generate.mockResolvedValue(generation);
  });

  it("reveals every remaining backend generation step immediately when reduced motion becomes active", async () => {
    const media = installMatchMedia(false);
    const user = userEvent.setup();
    renderPage();
    await screen.findByText("Local checkpoint loaded");

    await user.click(screen.getByRole("button", { name: "Generate" }));
    expect(await screen.findByText("Step 1")).toBeInTheDocument();
    expect(screen.queryByText("Step 2")).not.toBeInTheDocument();

    act(() => media.setMatches(true));
    await waitFor(() => expect(screen.getByText("Step 3")).toBeInTheDocument());
  });
});
