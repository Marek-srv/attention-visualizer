import { useEffect, useRef } from "react";

import type { TrainingStatus } from "../types/api";

type TrainingPollingOptions = {
  enabled: boolean;
  fetchStatus: (signal: AbortSignal) => Promise<TrainingStatus>;
  onStatus: (status: TrainingStatus) => void;
  onError: (error: unknown) => void;
  intervalMs?: number;
};

/** Polls only while a training job is running and cleans up both timer and request. */
export function useTrainingPolling({
  enabled,
  fetchStatus,
  onStatus,
  onError,
  intervalMs = 1200,
}: TrainingPollingOptions): void {
  const fetchRef = useRef(fetchStatus);
  const statusRef = useRef(onStatus);
  const errorRef = useRef(onError);

  useEffect(() => {
    fetchRef.current = fetchStatus;
    statusRef.current = onStatus;
    errorRef.current = onError;
  }, [fetchStatus, onError, onStatus]);

  useEffect(() => {
    if (!enabled) return undefined;

    let disposed = false;
    let timer: number | undefined;
    let controller: AbortController | undefined;

    const schedule = () => {
      timer = window.setTimeout(async () => {
        controller = new AbortController();
        try {
          const next = await fetchRef.current(controller.signal);
          if (disposed) return;
          statusRef.current(next);
          if (next.status === "running") schedule();
        } catch (error: unknown) {
          if (disposed || (error instanceof DOMException && error.name === "AbortError")) return;
          errorRef.current(error);
          schedule();
        }
      }, intervalMs);
    };

    schedule();
    return () => {
      disposed = true;
      controller?.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [enabled, intervalMs]);
}
