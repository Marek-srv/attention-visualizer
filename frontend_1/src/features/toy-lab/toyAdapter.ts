import type {
  FeedForwardParameters,
  NormalizationParameters,
  ToyInspectResponse,
  ToyWeights,
} from "../../types/api";
import type { ToyStageId } from "./toyState";

export type ToyStageDefinition = {
  id: ToyStageId;
  shortLabel: string;
  title: string;
  formula: string;
  available: (result: ToyInspectResponse) => boolean;
};

const always = (): boolean => true;
const never = (): boolean => false;

export const TOY_STAGES: readonly ToyStageDefinition[] = [
  { id: "text", shortLabel: "Text", title: "Input text", formula: String.raw`x = \text{input text}`, available: always },
  { id: "tokens", shortLabel: "Tokens", title: "Word and punctuation tokens", formula: String.raw`x \rightarrow (t_0,\ldots,t_n)`, available: always },
  { id: "embeddings", shortLabel: "Embedding", title: "Token and position embeddings", formula: String.raw`X_i = E(t_i) + P_i`, available: always },
  { id: "qkv", shortLabel: "Q · K · V", title: "Query, key, and value projections", formula: String.raw`Q=XW_Q,\quad K=XW_K,\quad V=XW_V`, available: always },
  { id: "attention", shortLabel: "Attention", title: "Scaled causal self-attention", formula: String.raw`A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)`, available: always },
  { id: "multi-head", shortLabel: "Multi-head", title: "Split, attend, concatenate, project", formula: String.raw`\operatorname{MHA}(X)=\operatorname{Concat}(H_1,H_2)W_O`, available: always },
  { id: "residual-norm", shortLabel: "Residual + Norm", title: "First residual connection and normalization", formula: String.raw`Y=\operatorname{LayerNorm}(X+\operatorname{MHA}(X))`, available: always },
  { id: "feed-forward", shortLabel: "FFN", title: "Feed-forward network", formula: String.raw`F=\operatorname{ReLU}(YW_1+b_1)W_2+b_2`, available: always },
  { id: "final-hidden", shortLabel: "Hidden state", title: "Transformer block output", formula: String.raw`H=\operatorname{LayerNorm}(Y+F)`, available: always },
  { id: "logits-softmax", shortLabel: "Logits", title: "Vocabulary logits and softmax", formula: String.raw`p=\operatorname{softmax}(HW_{vocab})`, available: never },
  { id: "next-token", shortLabel: "Next token", title: "Next-token selection", formula: String.raw`t_{n+1}\sim p(t\mid t_{\le n})`, available: never },
] as const;

export function stageDefinition(stage: ToyStageId): ToyStageDefinition {
  return TOY_STAGES.find((definition) => definition.id === stage) ?? TOY_STAGES[0]!;
}

export function stageVector(result: ToyInspectResponse, stage: ToyStageId, position: number): readonly number[] | null {
  const token = result.tokens[position];
  const projection = result.projections[position];
  const normalization = result.attention_sublayer.calculations[position];
  const feedForward = result.feed_forward_sublayer.calculations[position];
  switch (stage) {
    case "embeddings":
      return token?.combined_embedding ?? null;
    case "qkv":
      return projection?.query ?? null;
    case "attention":
      return result.attention.context_vectors[position] ?? null;
    case "multi-head":
      return result.multi_head_attention.projected_outputs[position] ?? null;
    case "residual-norm":
      return normalization?.layer_norm_output ?? null;
    case "feed-forward":
      return feedForward?.feed_forward_output ?? null;
    case "final-hidden":
      return feedForward?.transformer_block_output ?? null;
    default:
      return null;
  }
}

export function cloneToyWeights(weights: ToyWeights): ToyWeights {
  return {
    query: weights.query.map((row) => [...row]),
    key: weights.key.map((row) => [...row]),
    value: weights.value.map((row) => [...row]),
    output: weights.output.map((row) => [...row]),
  };
}

export function cloneNormalization(parameters: NormalizationParameters): NormalizationParameters {
  return { gamma: [...parameters.gamma], beta: [...parameters.beta], epsilon: parameters.epsilon };
}

export function feedForwardParametersFromResponse(result: ToyInspectResponse): FeedForwardParameters {
  const feedForward = result.feed_forward_sublayer;
  return {
    input_weights: feedForward.input_weight_matrix.map((row) => [...row]),
    input_bias: [...feedForward.input_bias],
    output_weights: feedForward.output_weight_matrix.map((row) => [...row]),
    output_bias: [...feedForward.output_bias],
    normalization: cloneNormalization(feedForward.normalization),
  };
}

export function cloneFeedForward(parameters: FeedForwardParameters): FeedForwardParameters {
  return {
    input_weights: parameters.input_weights.map((row) => [...row]),
    input_bias: [...parameters.input_bias],
    output_weights: parameters.output_weights.map((row) => [...row]),
    output_bias: [...parameters.output_bias],
    normalization: cloneNormalization(parameters.normalization),
  };
}

export function attentionCellIsMasked(result: ToyInspectResponse, query: number, key: number): boolean {
  return result.attention.causal_mask[query]?.[key] === false || result.attention.masked_scores[query]?.[key] === null;
}

export function approximatelyOne(value: number, tolerance = 0.001): boolean {
  return Math.abs(value - 1) <= tolerance;
}

export function isUnknownToken(token: ToyInspectResponse["tokens"][number]): boolean {
  return token.normalized === "<UNK>" || token.token_id === 0;
}
