import gsap from "gsap";
import { useEffect, useLayoutEffect, useReducer, useRef } from "react";

import type { ToyInspectResponse } from "../../types/api";
import { stageDefinition } from "../../features/toy-lab/toyAdapter";
import { initialToyExplainerState, toyExplainerReducer } from "../../features/toy-lab/toyState";
import TextbookPanel from "../textbook/TextbookPanel";
import GuidedControls from "./GuidedControls";
import MobileStageFlow from "./MobileStageFlow";
import StageDetail from "./StageDetail";
import TransformerCanvas from "./TransformerCanvas";
import { prefersReducedMotion } from "./useMobileLayout";
import useMobileLayout from "./useMobileLayout";
import styles from "./explainer.module.css";

type TransformerWorkspaceProps = {
  result: ToyInspectResponse;
  learningMode: "guided" | "explore";
  textbookOpen: boolean;
};

export default function TransformerWorkspace({ result, learningMode, textbookOpen }: TransformerWorkspaceProps) {
  const [state, dispatch] = useReducer(toyExplainerReducer, initialToyExplainerState);
  const detailRef = useRef<HTMLDivElement>(null);
  const mobile = useMobileLayout();

  useEffect(() => {
    dispatch({
      type: "clamp-selection",
      tokenCount: result.token_count,
      headCount: result.multi_head_attention.number_of_heads,
      dimensionCount: result.embedding_dimension,
      neuronCount: result.feed_forward_sublayer.hidden_dimension,
    });
  }, [result]);

  useEffect(() => {
    if (learningMode === "explore") dispatch({ type: "pause" });
  }, [learningMode]);

  useEffect(() => {
    if (!state.isPlaying) return undefined;
    const timeout = window.setTimeout(() => dispatch({ type: "next-stage" }), 1650 / state.speed);
    return () => window.clearTimeout(timeout);
  }, [state.isPlaying, state.speed, state.stage]);

  useEffect(() => {
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape" && state.expandedStage) dispatch({ type: "close-detail" });
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [state.expandedStage]);

  useLayoutEffect(() => {
    if (!detailRef.current || !state.expandedStage || prefersReducedMotion()) return undefined;
    const context = gsap.context(() => {
      gsap.fromTo(detailRef.current, { autoAlpha: 0, y: 16, scale: 0.99 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.34, ease: "power2.out" });
    }, detailRef);
    return () => context.revert();
  }, [state.expandedStage]);

  const activeStage = state.expandedStage ?? state.stage;
  const activeDefinition = stageDefinition(activeStage);

  return (
    <section className={styles.workspace} aria-label="Interactive Toy Transformer workspace">
      <div className={styles.workspaceMain}>
        {learningMode === "guided" ? <GuidedControls state={state} dispatch={dispatch} /> : (
          <p className={styles.exploreHint}>Explore mode: choose any token or stage. Your token selection persists as you move.</p>
        )}
        <div className={styles.tokenBar} role="group" aria-label="Keep one token selected across the pipeline">
          <span>Trace token</span>
          {result.tokens.map((token) => (
            <button
              type="button"
              key={token.position}
              aria-pressed={token.position === state.selectedTokenPosition}
              title={`${token.token}, ID ${token.token_id}, position ${token.position}`}
              onClick={() => dispatch({ type: "select-token", position: token.position })}
            >
              {token.token}<small>ID {token.token_id} · p{token.position}</small>
            </button>
          ))}
        </div>
        {mobile ? <MobileStageFlow result={result} state={state} dispatch={dispatch} /> : <TransformerCanvas result={result} state={state} dispatch={dispatch} />}
        {state.expandedStage ? (
          <div ref={detailRef} className={styles.detailAnimation}>
            <StageDetail result={result} state={state} dispatch={dispatch} onClose={() => dispatch({ type: "close-detail" })} />
          </div>
        ) : (
          <button type="button" className={styles.expandPrompt} onClick={() => dispatch({ type: "select-stage", stage: state.stage })}>
            Expand {activeDefinition.shortLabel} for exact values and arithmetic
          </button>
        )}
        {textbookOpen && mobile ? <TextbookPanel stage={activeStage} formula={activeDefinition.formula} mobile /> : null}
      </div>
      {textbookOpen && !mobile ? <TextbookPanel stage={activeStage} formula={activeDefinition.formula} /> : null}
    </section>
  );
}

