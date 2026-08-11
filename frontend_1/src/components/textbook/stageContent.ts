import type { ToyStageId } from "../../features/toy-lab/toyState";

export type StageLesson = {
  heading: string;
  does: string;
  values: string;
  notice: string;
  warning?: string;
};

export const STAGE_LESSONS: Record<ToyStageId, StageLesson> = {
  text: {
    heading: "Start with the exact input",
    does: "The backend receives this text and keeps punctuation as meaningful input.",
    values: "The character count includes spaces and punctuation.",
    notice: "A small text change can change token boundaries and every later calculation.",
  },
  tokens: {
    heading: "Split text into vocabulary items",
    does: "The transparent tokenizer separates words and punctuation, normalizes them, and looks up IDs.",
    values: "Each chip shows the original token, normalized form, vocabulary ID, and sequence position.",
    notice: "Position identifies where a token occurs; the ID identifies its vocabulary entry.",
    warning: "An unknown token is a fallback vocabulary item, not an error or a semantic judgment.",
  },
  embeddings: {
    heading: "Give each token a numeric starting state",
    does: "A token vector and a sinusoidal position vector are added dimension by dimension.",
    values: "The combined four-dimensional vector is the input X used by the projection stage.",
    notice: "Two identical tokens at different positions receive different combined vectors.",
  },
  qkv: {
    heading: "Create three views of each token",
    does: "The same combined vector is multiplied by separate query, key, and value matrices.",
    values: "Selecting an output dimension exposes every input-value × weight product used in that result.",
    notice: "Q, K, and V differ because their matrices differ, even though they start from the same X.",
  },
  attention: {
    heading: "Let each token read allowed earlier tokens",
    does: "Query-key scores are scaled, causally masked, normalized with softmax, and used to mix values.",
    values: "A matrix row belongs to one query token; each cell is how much of one key/value position it mixes.",
    notice: "Every valid attention row sums to approximately one, while future positions receive weight zero.",
    warning: "Attention weight is a mixing coefficient, not a complete measure of human-readable importance.",
  },
  "multi-head": {
    heading: "Read through multiple smaller channels",
    does: "The four projection dimensions split into two heads, attend independently, concatenate, and pass through WO.",
    values: "Each small heatmap is one head; the projected output returns to four model dimensions.",
    notice: "Heads can show different reading patterns because they use different dimension slices.",
    warning: "A visually strong head pattern alone does not explain the model's complete computation.",
  },
  "residual-norm": {
    heading: "Preserve the stream, then stabilize it",
    does: "The original combined embedding is added to the attention output before post-layer normalization.",
    values: "Mean and population variance are computed across the four residual dimensions for each token.",
    notice: "With default gamma and beta, the normalized values have mean near zero and variance near one.",
  },
  "feed-forward": {
    heading: "Transform each token independently",
    does: "A 4→8 projection, ReLU, and 8→4 projection transform every token before the second residual and normalization.",
    values: "Select a hidden neuron to inspect its weighted sum, bias, pre-activation, and active/inactive state.",
    notice: "Attention mixes positions; the feed-forward network mixes features within each token.",
    warning: "One active neuron is not automatically a single human-interpretable concept.",
  },
  "final-hidden": {
    heading: "Read the block's final token state",
    does: "The second normalized residual is the educational block output passed to a later prediction head.",
    values: "Four values are shown for each token because this Toy Math Lab uses d_model = 4.",
    notice: "The selected token remains selected so you can compare its state with earlier stages.",
  },
  "logits-softmax": {
    heading: "Project to vocabulary scores",
    does: "A trained language model would multiply the final hidden state by vocabulary weights, then normalize logits.",
    values: "The Toy inspection response does not include vocabulary logits, so this stage is intentionally unavailable here.",
    notice: "Use Predict or Trained Model Inspector for real checkpoint logits.",
    warning: "The interface never invents missing vocabulary scores.",
  },
  "next-token": {
    heading: "Choose a continuation",
    does: "A trained model can select or sample a token from the returned probability distribution.",
    values: "Toy Math Lab stops at the block output; next-token candidates come from the trained or real-model APIs.",
    notice: "Move to Predict to connect a saved checkpoint to an actual continuation.",
    warning: "Model probabilities describe the model's distribution, not factual confidence.",
  },
};

