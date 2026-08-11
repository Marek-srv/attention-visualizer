export const STAGE_IDS = [
  "text",
  "tokens",
  "embeddings",
  "qkv",
  "causal-attention",
  "multi-head-merge",
  "residual-norm",
  "feed-forward",
  "final-hidden",
  "logits-softmax",
  "next-token",
] as const;

export type StageId = (typeof STAGE_IDS)[number];

export type StageCategory = "input" | "transformer" | "output";

export interface StageDefinition {
  id: StageId;
  label: string;
  shortLabel: string;
  category: StageCategory;
  formula: string;
  notice: string;
}

export const STAGE_REGISTRY: readonly StageDefinition[] = [
  { id: "text", label: "Text", shortLabel: "Text", category: "input", formula: "x = \\text{prompt}", notice: "The prompt is the source sequence for this forward pass." },
  { id: "tokens", label: "Tokens", shortLabel: "Tokens", category: "input", formula: "x \\rightarrow (t_0, \\ldots, t_n)", notice: "Token IDs are vocabulary references, not numerical meaning." },
  { id: "embeddings", label: "Embeddings", shortLabel: "Embed", category: "transformer", formula: "X = E_{token} + E_{position}", notice: "Each token becomes a position-aware vector." },
  { id: "qkv", label: "Q, K, V", shortLabel: "QKV", category: "transformer", formula: "Q=XW_Q,\\ K=XW_K,\\ V=XW_V", notice: "Three projections prepare matching and carried information." },
  { id: "causal-attention", label: "Causal Attention", shortLabel: "Attention", category: "transformer", formula: "A=\\operatorname{softmax}(QK^T/\\sqrt{d_k}+M)", notice: "Future keys are masked before softmax." },
  { id: "multi-head-merge", label: "Multi-Head Merge", shortLabel: "Heads", category: "transformer", formula: "H=\\operatorname{Concat}(h_1,\\ldots,h_m)W_O", notice: "Heads read through separate learned projections before merging." },
  { id: "residual-norm", label: "Residual + Norm", shortLabel: "Residual", category: "transformer", formula: "Y=\\operatorname{LayerNorm}(X+H)", notice: "The residual stream preserves the incoming representation." },
  { id: "feed-forward", label: "Feed-Forward Network", shortLabel: "FFN", category: "transformer", formula: "F=\\sigma(YW_1+b_1)W_2+b_2", notice: "The FFN transforms each token position independently." },
  { id: "final-hidden", label: "Final Hidden State", shortLabel: "Hidden", category: "output", formula: "z=h_{last}", notice: "The final usable position summarizes the model state for prediction." },
  { id: "logits-softmax", label: "Logits + Softmax", shortLabel: "Softmax", category: "output", formula: "p=\\operatorname{softmax}(zW_{vocab}/T)", notice: "Probabilities are the model's distribution, not factual confidence." },
  { id: "next-token", label: "Next Token", shortLabel: "Token", category: "output", formula: "t_{next}\\sim p", notice: "Greedy decoding selects the maximum; sampling draws from the distribution." },
] as const;

const stageIndex = new Map<StageId, number>(STAGE_IDS.map((id, index) => [id, index]));

export function getStageIndex(stage: StageId): number {
  return stageIndex.get(stage) ?? 0;
}

export function getStageAt(index: number): StageId {
  const bounded = Math.min(Math.max(Math.trunc(index), 0), STAGE_IDS.length - 1);
  return STAGE_IDS[bounded] ?? STAGE_IDS[0];
}

export function getStageDefinition(stage: StageId): StageDefinition {
  return STAGE_REGISTRY[getStageIndex(stage)] ?? STAGE_REGISTRY[0]!;
}
