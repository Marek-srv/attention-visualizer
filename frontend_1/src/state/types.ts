import type { StageId } from "./stageRegistry";

export type WorkspaceMode = "toy" | "training" | "prediction" | "trained-inspector" | "real-model";
export type LearningMode = "guided" | "explore";
export type PlaybackSpeed = 0.5 | 1 | 1.5;

export interface PlaybackState {
  currentStageIndex: number;
  isPlaying: boolean;
  speed: PlaybackSpeed;
}

export interface ExplainerState {
  mode: WorkspaceMode;
  prompt: string;
  learningMode: LearningMode;
  selectedStage: StageId;
  expandedStage: StageId | null;
  selectedTokenIndex: number | null;
  selectedDimension: number | null;
  textbookOpen: boolean;
  playback: PlaybackState;
}

export type ExplainerAction =
  | { type: "set-mode"; mode: WorkspaceMode }
  | { type: "set-prompt"; prompt: string }
  | { type: "set-learning-mode"; mode: LearningMode }
  | { type: "select-stage"; stage: StageId }
  | { type: "expand-stage"; stage: StageId }
  | { type: "close-stage" }
  | { type: "select-token"; tokenIndex: number | null }
  | { type: "select-dimension"; dimension: number | null }
  | { type: "set-textbook-open"; open: boolean }
  | { type: "play" }
  | { type: "pause" }
  | { type: "toggle-playback" }
  | { type: "next-stage" }
  | { type: "previous-stage" }
  | { type: "restart-playback" }
  | { type: "set-playback-speed"; speed: PlaybackSpeed };
