import gsap from "gsap";
import { useLayoutEffect, useRef, type Dispatch, type ReactNode } from "react";
import { ArrowLeft, CircleSlash2, Split, Sigma } from "lucide-react";

import type { ToyInspectResponse } from "../../types/api";
import { attentionCellIsMasked, isUnknownToken, stageDefinition } from "../../features/toy-lab/toyAdapter";
import type { AttentionView, ProjectionKind, ToyExplainerAction, ToyExplainerState } from "../../features/toy-lab/toyState";
import AttentionArcs from "../visualizations/AttentionArcs";
import EmbeddingAdditionTable from "../visualizations/EmbeddingAdditionTable";
import Formula from "../visualizations/Formula";
import MatrixHeatmap from "../visualizations/MatrixHeatmap";
import MatrixTable from "../visualizations/MatrixTable";
import VectorRow from "../visualizations/VectorRow";
import FeedForwardNetwork from "../visualizations/FeedForwardNetwork";
import { prefersReducedMotion } from "./useMobileLayout";
import styles from "./explainer.module.css";

type StageDetailProps = {
  result: ToyInspectResponse;
  state: ToyExplainerState;
  dispatch: Dispatch<ToyExplainerAction>;
  onClose: () => void;
};

const projectionLabels: Record<ProjectionKind, string> = {
  query: "Q — what this token is looking for",
  key: "K — what this token can match",
  value: "V — what this token can pass forward",
};

const attentionLabels: Record<AttentionView, string> = {
  raw_scores: "QKᵀ raw scores",
  scaled_scores: "Scaled scores",
  masked_scores: "Causal mask",
  attention_weights: "Softmax weights",
  context_vectors: "Attention × V context",
};

function Selector({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className={styles.selector} role="group" aria-label={label}>
      <span>{label}</span>
      <div>{children}</div>
    </div>
  );
}

function TokenSelector({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  return (
    <Selector label="Inspect token">
      {result.tokens.map((token) => (
        <button
          type="button"
          key={token.position}
          aria-pressed={token.position === state.selectedTokenPosition}
          onClick={() => dispatch({ type: "select-token", position: token.position })}
        >
          {token.token}<small>p{token.position}</small>
        </button>
      ))}
    </Selector>
  );
}

function DimensionSelector({ count, state, dispatch, prefix = "d" }: { count: number; state: ToyExplainerState; dispatch: Dispatch<ToyExplainerAction>; prefix?: string }) {
  return (
    <Selector label="Inspect output dimension">
      {Array.from({ length: count }, (_, dimension) => (
        <button
          type="button"
          key={dimension}
          aria-pressed={dimension === state.selectedDimension}
          onClick={() => dispatch({ type: "select-dimension", dimension })}
        >
          {prefix}{dimension}
        </button>
      ))}
    </Selector>
  );
}

function TextDetail({ result }: { result: ToyInspectResponse }) {
  return (
    <div className={styles.detailStack}>
      <div className={styles.promptSample} aria-label={`Input text: ${result.text}`}>{result.text}</div>
      <dl className={styles.stats}>
        <div><dt>Characters</dt><dd>{result.character_count}</dd></div>
        <div><dt>Tokens</dt><dd>{result.token_count}</dd></div>
        <div><dt>Toy API phase</dt><dd>{result.phase}</dd></div>
      </dl>
      <p className={styles.explanation}>The run begins with the exact submitted text. Nothing in this view is generated or substituted.</p>
    </div>
  );
}

function TokensDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  return (
    <div className={styles.detailStack}>
      <div className={styles.tokenSplit} aria-label="Tokenization result">
        {result.tokens.map((token, index) => {
          const unknown = isUnknownToken(token);
          return (
            <div key={token.position} className={styles.tokenStep} data-toy-motion="token">
              {index > 0 ? <span aria-hidden="true">+</span> : null}
              <button
                type="button"
                data-selected={token.position === state.selectedTokenPosition ? "true" : "false"}
                onClick={() => dispatch({ type: "select-token", position: token.position })}
              >
                <strong>{token.token}</strong>
                <small>normalized: {token.normalized}</small>
                <small>ID {token.token_id} · position {token.position}</small>
                {unknown ? <em>Unknown token</em> : null}
              </button>
            </div>
          );
        })}
      </div>
      <p className={styles.explanation}>Words and punctuation become separate, reproducible vocabulary lookups. Selection stays attached to the token position across every later stage.</p>
    </div>
  );
}

function EmbeddingDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const token = result.tokens[state.selectedTokenPosition];
  if (!token) return null;
  const dimension = state.selectedDimension;
  const tokenValue = token.token_embedding[dimension] ?? 0;
  const positionValue = token.position_embedding[dimension] ?? 0;
  const combinedValue = token.combined_embedding[dimension] ?? 0;
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <div className={styles.operationFlow}>
        <span>Token embedding</span><b>+</b><span>Position embedding</span><b>=</b><span>Combined X</span>
      </div>
      <div data-toy-motion="embedding-part"><VectorRow label="Token E(t)" values={token.token_embedding} selectedDimension={dimension} onSelectDimension={(next) => dispatch({ type: "select-dimension", dimension: next })} /></div>
      <div data-toy-motion="embedding-part"><VectorRow label={`Position P(${token.position})`} values={token.position_embedding} selectedDimension={dimension} onSelectDimension={(next) => dispatch({ type: "select-dimension", dimension: next })} /></div>
      <div data-toy-motion="embedding-result"><VectorRow label="Combined X" values={token.combined_embedding} selectedDimension={dimension} tone="output" onSelectDimension={(next) => dispatch({ type: "select-dimension", dimension: next })} /></div>
      <EmbeddingAdditionTable
        tokenValues={token.token_embedding}
        positionValues={token.position_embedding}
        combinedValues={token.combined_embedding}
        selectedDimension={dimension}
        onSelectDimension={(next) => dispatch({ type: "select-dimension", dimension: next })}
      />
      <DimensionSelector count={result.embedding_dimension} state={state} dispatch={dispatch} />
      <Formula
        label={`Embedding addition at dimension ${dimension}`}
        expression={String.raw`X_{${dimension}}=${tokenValue.toFixed(4)}+${positionValue.toFixed(4)}=${combinedValue.toFixed(4)}`}
      />
    </div>
  );
}

function ProjectionDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const projection = result.projections[state.selectedTokenPosition];
  if (!projection) return null;
  const kind = state.selectedProjection;
  const vector = projection[kind];
  const breakdown = projection.breakdown[kind][state.selectedDimension];
  const matrix = result.weights[kind];
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <Selector label="Projection">
        {(["query", "key", "value"] as const).map((projectionKind) => (
          <button
            type="button"
            key={projectionKind}
            aria-pressed={projectionKind === kind}
            onClick={() => dispatch({ type: "select-projection", projection: projectionKind })}
          >
            {projectionKind[0]?.toUpperCase()}
          </button>
        ))}
      </Selector>
      <div className={styles.branchFlow}>
        <VectorRow label="Combined X" values={projection.input_vector} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} />
        <div className={styles.branchLabels}><span data-toy-motion="qkv-branch">× WQ → Q</span><span data-toy-motion="qkv-branch">× WK → K</span><span data-toy-motion="qkv-branch">× WV → V</span></div>
        <div data-toy-motion="qkv-branch"><VectorRow label={projectionLabels[kind]} values={vector} tone={kind} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      </div>
      <DimensionSelector count={result.embedding_dimension} state={state} dispatch={dispatch} prefix={kind[0]} />
      <MatrixTable title={`W${kind[0]?.toUpperCase()} — highlighted output column`} values={matrix} highlightedColumn={state.selectedDimension} />
      {breakdown ? (
        <div className={styles.calculation}>
          <h3>Every term contributing to output {state.selectedDimension}</h3>
          <div className={styles.terms}>
            {breakdown.terms.map((term) => (
              <span key={term.input_dimension}>
                x{term.input_dimension} {term.input_value.toFixed(4)} × w{term.input_dimension}{state.selectedDimension} {term.weight_value.toFixed(4)} = <strong>{term.product.toFixed(4)}</strong>
              </span>
            ))}
          </div>
          <Formula expression={String.raw`${kind[0]}_{${state.selectedDimension}}=${breakdown.terms.map((term) => term.product.toFixed(4)).join("+")}=${breakdown.result.toFixed(4)}`} />
        </div>
      ) : null}
    </div>
  );
}

function AttentionDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const attention = result.attention;
  const view = state.attentionView;
  const values: ReadonlyArray<ReadonlyArray<number | null>> = view === "context_vectors" ? attention.context_vectors : attention[view];
  const applyMask = view === "masked_scores" || view === "attention_weights";
  const contextView = view === "context_vectors";
  const query = state.selectedTokenPosition;
  const key = state.selectedKeyPosition;
  const calculation = attention.calculations.find((item) => item.query_position === query && item.key_position === key);
  const contextDimension = attention.context_calculations
    .find((item) => item.query_position === query)
    ?.dimensions[state.selectedDimension];
  const tokenLabels = result.tokens.map((token) => token.token);
  const masked = attentionCellIsMasked(result, query, key);
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <Selector label="Attention substage">
        {(Object.keys(attentionLabels) as AttentionView[]).map((attentionView) => (
          <button
            type="button"
            key={attentionView}
            aria-pressed={attentionView === view}
            onClick={() => dispatch({ type: "select-attention-view", view: attentionView })}
          >
            {attentionLabels[attentionView]}
          </button>
        ))}
      </Selector>
      <div className={styles.operationFlow}>
        <span>QKᵀ</span><b>→</b><span>÷ √{attention.key_dimension}</span><b>→</b><span>Causal mask</span><b>→</b><span>Softmax</span><b>→</b><span>× V</span>
      </div>
      <MatrixHeatmap
        title={attentionLabels[view]}
        values={values}
        rowLabels={tokenLabels}
        columnLabels={contextView ? Array.from({ length: result.embedding_dimension }, (_, dimension) => `d${dimension}`) : tokenLabels}
        allowedMask={applyMask ? attention.causal_mask : undefined}
        selected={{ row: query, column: contextView ? state.selectedDimension : key }}
        onSelect={({ row, column }) => {
          dispatch({ type: "select-token", position: row });
          dispatch(contextView ? { type: "select-dimension", dimension: column } : { type: "select-key", position: column });
        }}
        rowSums={view === "attention_weights" ? attention.row_sums : undefined}
      />
      {!contextView ? <AttentionArcs
        tokens={tokenLabels}
        queryPosition={query}
        weights={attention.attention_weights[query] ?? []}
        allowed={attention.causal_mask[query] ?? []}
        selectedKey={key}
        onSelectKey={(position) => dispatch({ type: "select-key", position })}
      /> : null}
      {calculation && !contextView ? (
        <div className={styles.calculation} aria-live="polite">
          <h3>{calculation.query_token} reads {calculation.key_token}</h3>
          {masked ? <p className={styles.maskedNotice}><CircleSlash2 aria-hidden="true" size={16} /> Masked future position · attention weight 0.0000</p> : null}
          <div className={styles.terms}>
            {calculation.products.map((term) => (
              <span key={term.dimension}>q{term.dimension} {term.query_value.toFixed(4)} × k{term.dimension} {term.key_value.toFixed(4)} = <strong>{term.product.toFixed(4)}</strong></span>
            ))}
          </div>
          <Formula expression={String.raw`s_{${query},${key}}=\frac{${calculation.raw_score.toFixed(4)}}{\sqrt{${attention.key_dimension}}}=${calculation.scaled_score.toFixed(4)},\quad a=${calculation.attention_weight.toFixed(4)}`} />
        </div>
      ) : null}
      <DimensionSelector count={result.embedding_dimension} state={state} dispatch={dispatch} prefix="z" />
      {contextDimension ? (
        <div className={styles.calculation}>
          <h3>Context dimension z{state.selectedDimension}</h3>
          <div className={styles.terms}>
            {contextDimension.terms.map((term) => (
              <span key={term.key_position}>{term.attention_weight.toFixed(4)} × V({term.key_token}) {term.value.toFixed(4)} = <strong>{term.product.toFixed(4)}</strong></span>
            ))}
          </div>
          <strong>Result: {contextDimension.result.toFixed(4)}</strong>
        </div>
      ) : null}
    </div>
  );
}

function MultiHeadDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const multiHead = result.multi_head_attention;
  const head = multiHead.heads[state.selectedHead];
  const position = state.selectedTokenPosition;
  const labels = result.tokens.map((token) => token.token);
  const outputCalculation = multiHead.output_calculations.find((item) => item.position === position)?.dimensions[state.selectedDimension];
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <div className={styles.operationFlow}><Split aria-hidden="true" size={17} /><span>Split dimensions</span><b>→</b><span>Attend per head</span><b>→</b><span>Concatenate</span><b>→</b><span>× WO</span></div>
      <Selector label="Select head">
        {multiHead.heads.map((item) => (
          <button type="button" key={item.head_index} aria-pressed={item.head_index === state.selectedHead} onClick={() => dispatch({ type: "select-head", head: item.head_index })}>
            Head {item.head_index + 1}<small>d{item.dimension_indices.join(", d")}</small>
          </button>
        ))}
      </Selector>
      <div className={styles.smallMultiples}>
        {multiHead.heads.map((item) => (
          <article key={item.head_index} data-selected={item.head_index === state.selectedHead ? "true" : "false"} data-toy-motion="head">
            <button type="button" aria-pressed={item.head_index === state.selectedHead} onClick={() => dispatch({ type: "select-head", head: item.head_index })}>Inspect Head {item.head_index + 1}</button>
            <MatrixHeatmap title={`Head ${item.head_index + 1}`} values={item.attention_weights} allowedMask={item.causal_mask} rowLabels={labels} columnLabels={labels} rowSums={item.row_sums} compact />
          </article>
        ))}
      </div>
      {head ? (
        <>
          <AttentionArcs tokens={labels} queryPosition={position} weights={head.attention_weights[position] ?? []} allowed={head.causal_mask[position] ?? []} selectedKey={state.selectedKeyPosition} onSelectKey={(next) => dispatch({ type: "select-key", position: next })} />
          <VectorRow label={`Head ${head.head_index + 1} context`} values={head.context_vectors[position] ?? []} />
        </>
      ) : null}
      <div data-toy-motion="head-merge"><VectorRow label="Concatenated heads" values={multiHead.concatenated_contexts[position] ?? []} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      <div data-toy-motion="head-merge"><VectorRow label="After WO" values={multiHead.projected_outputs[position] ?? []} tone="output" selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      <DimensionSelector count={multiHead.model_dimension} state={state} dispatch={dispatch} />
      <MatrixTable title="WO — output projection" values={multiHead.output_weight_matrix} highlightedColumn={state.selectedDimension} />
      {outputCalculation ? (
        <div className={styles.calculation}>
          <h3>WO output dimension {state.selectedDimension}</h3>
          <div className={styles.terms}>{outputCalculation.terms.map((term) => <span key={term.input_dimension}>{term.concatenated_input_value.toFixed(4)} × {term.weight_value.toFixed(4)} = <strong>{term.product.toFixed(4)}</strong></span>)}</div>
          <strong>Result: {outputCalculation.result.toFixed(4)}</strong>
        </div>
      ) : null}
    </div>
  );
}

function ResidualNormDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const calculation = result.attention_sublayer.calculations[state.selectedTokenPosition];
  if (!calculation) return null;
  const term = calculation.normalization_terms[state.selectedDimension];
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <div className={styles.operationFlow}><span>Original X</span><b>+</b><span>Attention contribution</span><b>=</b><span>Residual</span><b>→</b><span>LayerNorm</span></div>
      <div data-toy-motion="residual-input"><VectorRow label="Original X" values={calculation.input_vector} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      <div data-toy-motion="residual-contribution"><VectorRow label="Attention output" values={calculation.attention_output} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      <div data-toy-motion="residual-result"><VectorRow label="Residual" values={calculation.residual_vector} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} /></div>
      <VectorRow label="Normalized" values={calculation.normalized_vector} selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} />
      <VectorRow label="Gamma · norm + beta" values={calculation.layer_norm_output} tone="output" selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} />
      <dl className={styles.stats}>
        <div><dt>Mean</dt><dd>{calculation.mean.toFixed(4)}</dd></div>
        <div><dt>Population variance</dt><dd>{calculation.variance.toFixed(4)}</dd></div>
        <div><dt>Epsilon</dt><dd>{calculation.epsilon.toFixed(5)}</dd></div>
        <div><dt>Standard deviation</dt><dd>{calculation.standard_deviation.toFixed(4)}</dd></div>
      </dl>
      <DimensionSelector count={result.embedding_dimension} state={state} dispatch={dispatch} />
      {term ? <Formula expression={String.raw`y_{${state.selectedDimension}}=${term.gamma.toFixed(4)}\left(\frac{${term.residual_value.toFixed(4)}-${term.mean.toFixed(4)}}{${term.standard_deviation.toFixed(4)}}\right)+${term.beta.toFixed(4)}=${term.output_value.toFixed(4)}`} /> : null}
    </div>
  );
}

function FeedForwardDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  const feedForward = result.feed_forward_sublayer;
  const calculation = feedForward.calculations[state.selectedTokenPosition];
  if (!calculation) return null;
  const neuron = calculation.hidden_calculations[state.selectedNeuron];
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <div className={styles.operationFlow}><span>4 features</span><b>× W1 + b1</b><span>8 hidden neurons</span><b>{feedForward.activation}</b><span>× W2 + b2</span><b>→</b><span>4 features</span></div>
      <VectorRow label="Input Y" values={calculation.input_vector} />
      <FeedForwardNetwork
        input={calculation.input_vector}
        hiddenPreActivation={calculation.pre_activation_vector}
        hiddenActivation={calculation.activated_vector}
        output={calculation.feed_forward_output}
        activationName={feedForward.activation}
        selectedNeuron={state.selectedNeuron}
        onSelectNeuron={(neuronIndex) => dispatch({ type: "select-neuron", neuron: neuronIndex })}
      />
      <VectorRow label="Pre-activation" values={calculation.pre_activation_vector} selectedDimension={state.selectedNeuron} onSelectDimension={(neuronIndex) => dispatch({ type: "select-neuron", neuron: neuronIndex })} />
      <VectorRow label={`${feedForward.activation} activation`} values={calculation.activated_vector} selectedDimension={state.selectedNeuron} onSelectDimension={(neuronIndex) => dispatch({ type: "select-neuron", neuron: neuronIndex })} />
      <Selector label="Inspect hidden neuron">
        {calculation.hidden_calculations.map((item) => (
          <button type="button" key={item.hidden_neuron} aria-pressed={item.hidden_neuron === state.selectedNeuron} onClick={() => dispatch({ type: "select-neuron", neuron: item.hidden_neuron })}>
            h{item.hidden_neuron}<small>{item.is_active ? "Active" : "Inactive"}</small>
          </button>
        ))}
      </Selector>
      {neuron ? (
        <div className={styles.calculation}>
          <h3>Neuron h{neuron.hidden_neuron}: {neuron.is_active ? "Active" : "Inactive"}</h3>
          <div className={styles.terms}>{neuron.terms.map((term) => <span key={term.input_dimension}>x{term.input_dimension} {term.input_value.toFixed(4)} × {term.weight_value.toFixed(4)} = <strong>{term.product.toFixed(4)}</strong></span>)}</div>
          <Formula expression={String.raw`h_{${neuron.hidden_neuron}}=\operatorname{${feedForward.activation}}(${neuron.weighted_sum.toFixed(4)}+${neuron.bias.toFixed(4)})=${neuron.activated.toFixed(4)}`} />
          <p>A single active neuron is not automatically a human-interpretable feature.</p>
        </div>
      ) : null}
      <VectorRow label="FFN output" values={calculation.feed_forward_output} />
      <VectorRow label="Second residual" values={calculation.residual_vector} />
      <VectorRow label="Block output" values={calculation.transformer_block_output} tone="output" />
    </div>
  );
}

function FinalHiddenDetail({ result, state, dispatch }: Omit<StageDetailProps, "onClose">) {
  return (
    <div className={styles.detailStack}>
      <TokenSelector result={result} state={state} dispatch={dispatch} />
      <VectorRow label="Final hidden state H" values={result.feed_forward_sublayer.transformer_block_outputs[state.selectedTokenPosition] ?? []} tone="output" selectedDimension={state.selectedDimension} onSelectDimension={(dimension) => dispatch({ type: "select-dimension", dimension })} />
      <p className={styles.explanation}>This is the final value returned by the Toy Transformer block. A trained vocabulary projection is intentionally outside the toy endpoint.</p>
    </div>
  );
}

function UnavailableDetail({ result, stage }: { result: ToyInspectResponse; stage: "logits-softmax" | "next-token" }) {
  return (
    <div className={styles.unavailable}>
      <Sigma aria-hidden="true" size={28} />
      <h3>{stage === "logits-softmax" ? "Vocabulary logits are not part of /api/inspect" : "Toy Math Lab does not predict a token"}</h3>
      <p>The successful response ends with {result.feed_forward_sublayer.transformer_block_outputs.length} final hidden-state vector{result.token_count === 1 ? "" : "s"}. Use the Predict mode with a loaded local checkpoint for real logits, probabilities, and generation.</p>
    </div>
  );
}

export default function StageDetail({ result, state, dispatch, onClose }: StageDetailProps) {
  const stage = state.expandedStage ?? state.stage;
  const definition = stageDefinition(stage);
  const detailRef = useRef<HTMLElement>(null);
  useLayoutEffect(() => {
    if (!detailRef.current || prefersReducedMotion()) return undefined;
    const context = gsap.context(() => {
      const timeline = gsap.timeline({ delay: 0.16, defaults: { duration: 0.32, ease: "power2.out" } });
      switch (stage) {
        case "tokens":
          timeline.fromTo('[data-toy-motion="token"]', { autoAlpha: 0, x: -16 }, { autoAlpha: 1, x: 0, stagger: 0.07 });
          break;
        case "embeddings":
          timeline
            .fromTo('[data-toy-motion="embedding-part"]', { autoAlpha: 0, x: -18 }, { autoAlpha: 1, x: 0, stagger: 0.08 })
            .fromTo('[data-toy-motion="embedding-result"]', { autoAlpha: 0, scale: 0.9 }, { autoAlpha: 1, scale: 1 }, "-=0.12");
          break;
        case "qkv":
          timeline.fromTo('[data-toy-motion="qkv-branch"]', { autoAlpha: 0, scaleX: 0.35, transformOrigin: "left center" }, { autoAlpha: 1, scaleX: 1, stagger: 0.07 });
          break;
        case "multi-head":
          timeline
            .fromTo('[data-toy-motion="head"]', { autoAlpha: 0, y: 14 }, { autoAlpha: 1, y: 0, stagger: 0.1 })
            .fromTo('[data-toy-motion="head-merge"]', { autoAlpha: 0, scaleX: 0.65 }, { autoAlpha: 1, scaleX: 1 }, "-=0.08");
          break;
        case "residual-norm":
          timeline
            .fromTo('[data-toy-motion="residual-input"]', { autoAlpha: 0, x: -18 }, { autoAlpha: 1, x: 0 })
            .fromTo('[data-toy-motion="residual-contribution"]', { autoAlpha: 0, x: 18 }, { autoAlpha: 1, x: 0 }, "<")
            .fromTo('[data-toy-motion="residual-result"]', { autoAlpha: 0, scale: 0.9 }, { autoAlpha: 1, scale: 1 });
          break;
        default:
          break;
      }
    }, detailRef);
    return () => context.revert();
  }, [stage, result]);
  let detail: ReactNode;
  switch (stage) {
    case "text": detail = <TextDetail result={result} />; break;
    case "tokens": detail = <TokensDetail result={result} state={state} dispatch={dispatch} />; break;
    case "embeddings": detail = <EmbeddingDetail result={result} state={state} dispatch={dispatch} />; break;
    case "qkv": detail = <ProjectionDetail result={result} state={state} dispatch={dispatch} />; break;
    case "attention": detail = <AttentionDetail result={result} state={state} dispatch={dispatch} />; break;
    case "multi-head": detail = <MultiHeadDetail result={result} state={state} dispatch={dispatch} />; break;
    case "residual-norm": detail = <ResidualNormDetail result={result} state={state} dispatch={dispatch} />; break;
    case "feed-forward": detail = <FeedForwardDetail result={result} state={state} dispatch={dispatch} />; break;
    case "final-hidden": detail = <FinalHiddenDetail result={result} state={state} dispatch={dispatch} />; break;
    case "logits-softmax": detail = <UnavailableDetail result={result} stage="logits-softmax" />; break;
    case "next-token": detail = <UnavailableDetail result={result} stage="next-token" />; break;
  }
  return (
    <article ref={detailRef} className={styles.detail} aria-labelledby="expanded-stage-title">
      <header>
        <div>
          <span>Expanded stage</span>
          <h2 id="expanded-stage-title">{definition.title}</h2>
        </div>
        <button type="button" className={styles.backButton} onClick={onClose}><ArrowLeft aria-hidden="true" size={16} /> Back to model</button>
      </header>
      <Formula expression={definition.formula} label={`${definition.title} formula`} />
      {detail}
    </article>
  );
}
