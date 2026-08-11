import { useEffect } from "react";

import type { PlaybackSpeed } from "../state/types";

export interface GuidedPlaybackOptions {
  isPlaying: boolean;
  currentStageIndex: number;
  stageCount: number;
  speed: PlaybackSpeed;
  onAdvance: () => void;
  onStop: () => void;
  baseDelay?: number;
}

export function playbackDelay(baseDelay: number, speed: PlaybackSpeed): number {
  return Math.max(0, baseDelay) / speed;
}

export function useGuidedPlayback({
  isPlaying,
  currentStageIndex,
  stageCount,
  speed,
  onAdvance,
  onStop,
  baseDelay = 1800,
}: GuidedPlaybackOptions): void {
  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }
    if (stageCount <= 0 || currentStageIndex >= stageCount - 1) {
      onStop();
      return undefined;
    }
    const timer = window.setTimeout(onAdvance, playbackDelay(baseDelay, speed));
    return () => window.clearTimeout(timer);
  }, [baseDelay, currentStageIndex, isPlaying, onAdvance, onStop, speed, stageCount]);
}
