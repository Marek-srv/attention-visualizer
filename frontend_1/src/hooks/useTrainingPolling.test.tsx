import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { TrainingStatus } from "../types/api";
import { useTrainingPolling } from "./useTrainingPolling";

function trainingStatus(status: TrainingStatus["status"]): TrainingStatus {
  return {
    state: status,
    status,
    job_id: status === "idle" ? null : "job-1",
    current_epoch: status === "idle" ? 0 : 1,
    total_epochs: status === "idle" ? 0 : 3,
    latest_completed_epoch: status === "idle" ? 0 : 1,
    latest_metrics: null,
    history: [],
    best_validation_loss: null,
    cancellation_requested: false,
    checkpoint_file: "tiny_transformer_best.pt",
    checkpoint_available: false,
    error: null,
    model_config: null,
    training_config: null,
  };
}

describe("useTrainingPolling", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("starts only when enabled and stops after a terminal status", async () => {
    const fetchStatus = vi
      .fn<(signal: AbortSignal) => Promise<TrainingStatus>>()
      .mockResolvedValueOnce(trainingStatus("running"))
      .mockResolvedValueOnce(trainingStatus("completed"));
    const onStatus = vi.fn();
    const onError = vi.fn();
    const { rerender } = renderHook(
      ({ enabled }) => useTrainingPolling({ enabled, fetchStatus, onStatus, onError, intervalMs: 100 }),
      { initialProps: { enabled: false } },
    );

    await act(async () => vi.advanceTimersByTimeAsync(300));
    expect(fetchStatus).not.toHaveBeenCalled();

    rerender({ enabled: true });
    await act(async () => vi.advanceTimersByTimeAsync(100));
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ status: "running" }));

    await act(async () => vi.advanceTimersByTimeAsync(100));
    expect(fetchStatus).toHaveBeenCalledTimes(2);
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ status: "completed" }));

    await act(async () => vi.advanceTimersByTimeAsync(500));
    expect(fetchStatus).toHaveBeenCalledTimes(2);
    expect(onError).not.toHaveBeenCalled();
  });

  it("aborts an in-flight status request when unmounted", async () => {
    let observedSignal: AbortSignal | undefined;
    const fetchStatus = vi.fn((signal: AbortSignal) => {
      observedSignal = signal;
      return new Promise<TrainingStatus>(() => undefined);
    });
    const { unmount } = renderHook(() => useTrainingPolling({
      enabled: true,
      fetchStatus,
      onStatus: vi.fn(),
      onError: vi.fn(),
      intervalMs: 100,
    }));

    await act(async () => vi.advanceTimersByTimeAsync(100));
    expect(observedSignal?.aborted).toBe(false);
    unmount();
    expect(observedSignal?.aborted).toBe(true);
  });
});
