import { getStageAt, getStageIndex, STAGE_IDS } from "./stageRegistry";
import type { ExplainerAction, ExplainerState } from "./types";

export function createInitialExplainerState(): ExplainerState {
  return {
    mode: "toy",
    prompt: "I love",
    learningMode: "guided",
    selectedStage: STAGE_IDS[0],
    expandedStage: null,
    selectedTokenIndex: null,
    selectedDimension: null,
    textbookOpen: true,
    playback: {
      currentStageIndex: 0,
      isPlaying: false,
      speed: 1,
    },
  };
}

export const initialExplainerState = createInitialExplainerState();

export function explainerReducer(state: ExplainerState, action: ExplainerAction): ExplainerState {
  switch (action.type) {
    case "set-mode":
      return {
        ...state,
        mode: action.mode,
        expandedStage: null,
        playback: { ...state.playback, isPlaying: false },
      };
    case "set-prompt":
      return { ...state, prompt: action.prompt };
    case "set-learning-mode":
      return {
        ...state,
        learningMode: action.mode,
        playback: { ...state.playback, isPlaying: action.mode === "guided" && state.playback.isPlaying },
      };
    case "select-stage": {
      const currentStageIndex = getStageIndex(action.stage);
      return {
        ...state,
        selectedStage: action.stage,
        playback: { ...state.playback, currentStageIndex },
      };
    }
    case "expand-stage": {
      const closing = state.expandedStage === action.stage;
      const currentStageIndex = getStageIndex(action.stage);
      return {
        ...state,
        selectedStage: action.stage,
        expandedStage: closing ? null : action.stage,
        playback: { ...state.playback, currentStageIndex },
      };
    }
    case "close-stage":
      return { ...state, expandedStage: null };
    case "select-token":
      return { ...state, selectedTokenIndex: action.tokenIndex };
    case "select-dimension":
      return { ...state, selectedDimension: action.dimension };
    case "set-textbook-open":
      return { ...state, textbookOpen: action.open };
    case "play":
      return state.playback.currentStageIndex >= STAGE_IDS.length - 1
        ? state
        : { ...state, playback: { ...state.playback, isPlaying: true } };
    case "pause":
      return { ...state, playback: { ...state.playback, isPlaying: false } };
    case "toggle-playback":
      return state.playback.currentStageIndex >= STAGE_IDS.length - 1
        ? { ...state, playback: { ...state.playback, isPlaying: false } }
        : { ...state, playback: { ...state.playback, isPlaying: !state.playback.isPlaying } };
    case "next-stage": {
      const currentStageIndex = Math.min(state.playback.currentStageIndex + 1, STAGE_IDS.length - 1);
      const atFinalStage = currentStageIndex === STAGE_IDS.length - 1;
      return {
        ...state,
        selectedStage: getStageAt(currentStageIndex),
        playback: { ...state.playback, currentStageIndex, isPlaying: atFinalStage ? false : state.playback.isPlaying },
      };
    }
    case "previous-stage": {
      const currentStageIndex = Math.max(state.playback.currentStageIndex - 1, 0);
      return {
        ...state,
        selectedStage: getStageAt(currentStageIndex),
        playback: { ...state.playback, currentStageIndex },
      };
    }
    case "restart-playback":
      return {
        ...state,
        selectedStage: STAGE_IDS[0],
        expandedStage: null,
        playback: { ...state.playback, currentStageIndex: 0, isPlaying: false },
      };
    case "set-playback-speed":
      return { ...state, playback: { ...state.playback, speed: action.speed } };
  }
}
