import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../../api/endpoints";
import { isAbortError } from "../../api/errors";
import type { ToyInspectRequest, ToyInspectResponse } from "../../types/api";

type ToyInspectionState = {
  result: ToyInspectResponse | null;
  defaults: ToyInspectResponse | null;
  loading: boolean;
  error: string | null;
  revision: number;
};

export default function useToyInspection(prompt: string, runNonce: number) {
  const [state, setState] = useState<ToyInspectionState>({ result: null, defaults: null, loading: false, error: null, revision: 0 });
  const controllerRef = useRef<AbortController | null>(null);
  const promptRef = useRef(prompt);

  const inspect = useCallback(async (payload: ToyInspectRequest, captureDefaults = false): Promise<ToyInspectResponse | null> => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const response = await api.inspectToy(payload, { signal: controller.signal });
      setState((current) => ({
        result: response,
        defaults: captureDefaults && current.defaults === null ? response : current.defaults,
        loading: false,
        error: null,
        revision: current.revision + 1,
      }));
      return response;
    } catch (error) {
      if (isAbortError(error)) return null;
      const message = error instanceof Error ? error.message : "The Toy inspection request failed.";
      setState((current) => ({ ...current, loading: false, error: message }));
      return null;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }, []);

  useEffect(() => {
    promptRef.current = prompt;
  }, [prompt]);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      void inspect({ text: promptRef.current.trim() || "I love" }, true);
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [inspect, runNonce]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  return { ...state, inspect };
}
