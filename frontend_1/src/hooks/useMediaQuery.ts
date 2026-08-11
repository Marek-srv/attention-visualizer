import { useCallback, useMemo, useSyncExternalStore } from "react";

export function useMediaQuery(query: string, fallback = false): boolean {
  const mediaQuery = useMemo(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return null;
    }
    return window.matchMedia(query);
  }, [query]);

  const subscribe = useCallback((notify: () => void) => {
    if (!mediaQuery) {
      return () => undefined;
    }
    mediaQuery.addEventListener("change", notify);
    return () => mediaQuery.removeEventListener("change", notify);
  }, [mediaQuery]);

  const getSnapshot = useCallback(() => mediaQuery?.matches ?? fallback, [fallback, mediaQuery]);
  const getServerSnapshot = useCallback(() => fallback, [fallback]);

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

export function useMobileLayout(): boolean {
  return useMediaQuery("(max-width: 720px)");
}
