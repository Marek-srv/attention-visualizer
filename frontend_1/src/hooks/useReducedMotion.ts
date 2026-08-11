import { useMediaQuery } from "./useMediaQuery";

export const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";

export function usePrefersReducedMotion(): boolean {
  return useMediaQuery(REDUCED_MOTION_QUERY);
}

export function motionDuration(milliseconds: number, reducedMotion: boolean): number {
  return reducedMotion ? 0 : Math.max(0, milliseconds);
}
