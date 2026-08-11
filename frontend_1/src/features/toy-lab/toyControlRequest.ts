import type {
  FeedForwardParameters,
  NormalizationParameters,
  ToyInspectRequest,
  ToyWeightsInput,
} from "../../types/api";
import { cloneFeedForward, cloneNormalization } from "./toyAdapter";

export type ToyControlValues = {
  weights: ToyWeightsInput;
  normalization: NormalizationParameters;
  feedForward: FeedForwardParameters;
};

export function toInspectRequest(text: string, values: ToyControlValues): ToyInspectRequest {
  return {
    text,
    weights: values.weights,
    normalization: cloneNormalization(values.normalization),
    feed_forward: cloneFeedForward(values.feedForward),
  };
}
