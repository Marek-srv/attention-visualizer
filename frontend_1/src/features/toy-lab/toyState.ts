export const TOY_STAGE_IDS = [
  "text",
  "tokens",
  "embeddings",
  "qkv",
  "attention",
  "multi-head",
  "residual-norm",
  "feed-forward",
  "final-hidden",
  "logits-softmax",
  "next-token",
] as const;

export type ToyStageId = (typeof TOY_STAGE_IDS)[number];
export type ProjectionKind = "query" | "key" | "value";
export type AttentionView = "raw_scores" | "scaled_scores" | "masked_scores" | "attention_weights" | "context_vectors";

export type ToyExplainerState = {
  stage: ToyStageId;
  expandedStage: ToyStageId | null;
  selectedTokenPosition: number;
  selectedKeyPosition: number;
  selectedDimension: number;
  selectedHead: number;
  selectedProjection: ProjectionKind;
  attentionView: AttentionView;
  selectedNeuron: number;
  isPlaying: boolean;
  speed: 0.5 | 1 | 1.5;
};

export type ToyExplainerAction =
  | { type: "select-stage"; stage: ToyStageId; expand?: boolean }
  | { type: "close-detail" }
  | { type: "select-token"; position: number }
  | { type: "select-key"; position: number }
  | { type: "select-dimension"; dimension: number }
  | { type: "select-head"; head: number }
  | { type: "select-projection"; projection: ProjectionKind }
  | { type: "select-attention-view"; view: AttentionView }
  | { type: "select-neuron"; neuron: number }
  | { type: "previous-stage" }
  | { type: "next-stage" }
  | { type: "restart" }
  | { type: "play" }
  | { type: "pause" }
  | { type: "set-speed"; speed: 0.5 | 1 | 1.5 }
  | { type: "clamp-selection"; tokenCount: number; headCount: number; dimensionCount: number; neuronCount: number };

export const initialToyExplainerState: ToyExplainerState = {
  stage: "text",
  expandedStage: null,
  selectedTokenPosition: 0,
  selectedKeyPosition: 0,
  selectedDimension: 0,
  selectedHead: 0,
  selectedProjection: "query",
  attentionView: "attention_weights",
  selectedNeuron: 0,
  isPlaying: false,
  speed: 1,
};

function clamp(value: number, maximumExclusive: number): number {
  return Math.max(0, Math.min(value, Math.max(maximumExclusive - 1, 0)));
}

function moveStage(state: ToyExplainerState, delta: -1 | 1): ToyExplainerState {
  const currentIndex = TOY_STAGE_IDS.indexOf(state.stage);
  const nextIndex = clamp(currentIndex + delta, TOY_STAGE_IDS.length);
  const atEnd = nextIndex === TOY_STAGE_IDS.length - 1;
  return {
    ...state,
    stage: TOY_STAGE_IDS[nextIndex] ?? state.stage,
    expandedStage: state.expandedStage === null ? null : TOY_STAGE_IDS[nextIndex] ?? state.stage,
    isPlaying: atEnd ? false : state.isPlaying,
  };
}

export function toyExplainerReducer(state: ToyExplainerState, action: ToyExplainerAction): ToyExplainerState {
  switch (action.type) {
    case "select-stage":
      return {
        ...state,
        stage: action.stage,
        expandedStage: action.expand === false || state.expandedStage === action.stage ? null : action.stage,
        isPlaying: false,
      };
    case "close-detail":
      return { ...state, expandedStage: null };
    case "select-token":
      return { ...state, selectedTokenPosition: Math.max(0, action.position) };
    case "select-key":
      return { ...state, selectedKeyPosition: Math.max(0, action.position) };
    case "select-dimension":
      return { ...state, selectedDimension: Math.max(0, action.dimension) };
    case "select-head":
      return { ...state, selectedHead: Math.max(0, action.head) };
    case "select-projection":
      return { ...state, selectedProjection: action.projection };
    case "select-attention-view":
      return { ...state, attentionView: action.view };
    case "select-neuron":
      return { ...state, selectedNeuron: Math.max(0, action.neuron) };
    case "previous-stage":
      return moveStage(state, -1);
    case "next-stage":
      return moveStage(state, 1);
    case "restart":
      return { ...state, stage: "text", expandedStage: null, isPlaying: false };
    case "play":
      return { ...state, isPlaying: state.stage !== "next-token" };
    case "pause":
      return { ...state, isPlaying: false };
    case "set-speed":
      return { ...state, speed: action.speed };
    case "clamp-selection":
      return {
        ...state,
        selectedTokenPosition: clamp(state.selectedTokenPosition, action.tokenCount),
        selectedKeyPosition: clamp(state.selectedKeyPosition, action.tokenCount),
        selectedHead: clamp(state.selectedHead, action.headCount),
        selectedDimension: clamp(state.selectedDimension, action.dimensionCount),
        selectedNeuron: clamp(state.selectedNeuron, action.neuronCount),
      };
  }
}

export function stageProgress(stage: ToyStageId): number {
  return TOY_STAGE_IDS.indexOf(stage) / (TOY_STAGE_IDS.length - 1);
}
