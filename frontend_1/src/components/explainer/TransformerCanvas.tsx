import { curveBumpX, line, max, scaleLinear } from "d3";
import { Minus, Plus, RotateCcw } from "lucide-react";
import { useMemo, useRef, useState, type Dispatch, type KeyboardEvent, type PointerEvent } from "react";

import type { ToyInspectResponse } from "../../types/api";
import { stageVector, TOY_STAGES } from "../../features/toy-lab/toyAdapter";
import type { ToyExplainerAction, ToyExplainerState, ToyStageId } from "../../features/toy-lab/toyState";
import styles from "./explainer.module.css";

type TransformerCanvasProps = {
  result: ToyInspectResponse;
  state: ToyExplainerState;
  dispatch: Dispatch<ToyExplainerAction>;
};

const VIEW_WIDTH = 1580;
const STAGE_TOP = 46;
const STAGE_WIDTH = 116;
const STAGE_GAP = 25;
const ROW_HEIGHT = 48;

type HoveredMark = { stage: ToyStageId; position: number } | null;

export default function TransformerCanvas({ result, state, dispatch }: TransformerCanvasProps) {
  const [zoom, setZoom] = useState(1);
  const [offsetX, setOffsetX] = useState(0);
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [hovered, setHovered] = useState<HoveredMark>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const stageHeight = Math.max(result.token_count * ROW_HEIGHT + 72, 180);
  const viewHeight = stageHeight + 110;
  const xForStage = (index: number): number => 34 + index * (STAGE_WIDTH + STAGE_GAP);
  const yForToken = (position: number): number => STAGE_TOP + 72 + position * ROW_HEIGHT;

  const color = useMemo(() => {
    const allValues = TOY_STAGES.flatMap((stage) => result.tokens.flatMap((token) => stageVector(result, stage.id, token.position) ?? []));
    const extent = Math.max(max(allValues, (value) => Math.abs(value)) ?? 1, 1e-8);
    return scaleLinear<string>().domain([-extent, 0, extent]).range(["#d86f68", "#15312a", "#62dda7"]).clamp(true);
  }, [result]);

  const path = useMemo(
    () => line<{ x: number; y: number }>().x((point) => point.x).y((point) => point.y).curve(curveBumpX),
    [],
  );

  const selectStage = (stage: ToyStageId): void => dispatch({ type: "select-stage", stage });
  const onStageKeyDown = (event: KeyboardEvent<SVGGElement>, stage: ToyStageId): void => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectStage(stage);
    }
  };
  const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragStart(event.clientX - offsetX);
  };
  const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
    if (dragStart === null) return;
    setOffsetX(Math.max(-620, Math.min(180, event.clientX - dragStart)));
  };
  const onPointerUp = (event: PointerEvent<SVGSVGElement>): void => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDragStart(null);
  };

  return (
    <div className={styles.canvasRegion}>
      <div className={styles.canvasToolbar}>
        <p><span className={styles.flowSwatch} /> Ordinary path width shows data flow only; it does not encode magnitude.</p>
        <div role="group" aria-label="Model view controls">
          <button type="button" aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.72, value - 0.12))}><Minus aria-hidden="true" size={15} /></button>
          <output aria-label="Current zoom">{Math.round(zoom * 100)}%</output>
          <button type="button" aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(1.8, value + 0.12))}><Plus aria-hidden="true" size={15} /></button>
          <button type="button" onClick={() => { setZoom(1); setOffsetX(0); }}><RotateCcw aria-hidden="true" size={14} /> Reset view</button>
        </div>
      </div>
      {hovered ? (
        <output className={styles.canvasTooltip} aria-live="polite">
          {result.tokens[hovered.position]?.token} · ID {result.tokens[hovered.position]?.token_id} · position {hovered.position} · {TOY_STAGES.find((stage) => stage.id === hovered.stage)?.shortLabel}
        </output>
      ) : null}
      <svg
        ref={svgRef}
        className={styles.canvas}
        viewBox={`0 0 ${VIEW_WIDTH} ${viewHeight}`}
        role="img"
        aria-labelledby="transformer-canvas-title transformer-canvas-description"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={() => setDragStart(null)}
      >
        <title id="transformer-canvas-title">Connected Toy Transformer forward pass</title>
        <desc id="transformer-canvas-description">Token rows move from text through the Transformer block. Select a token or stage to inspect exact backend values.</desc>
        <g transform={`translate(${offsetX} 0) scale(${zoom})`}>
          <g className={styles.flowPaths} aria-hidden="true">
            {result.tokens.flatMap((token) => TOY_STAGES.slice(0, -1).map((stage, stageIndex) => {
              const nextStage = TOY_STAGES[stageIndex + 1];
              if (!nextStage) return null;
              const y = yForToken(token.position) + 14;
              const d = path([{ x: xForStage(stageIndex) + STAGE_WIDTH, y }, { x: xForStage(stageIndex + 1), y }]);
              return (
                <path
                  key={`${token.position}-${stage.id}`}
                  d={d ?? undefined}
                  fill="none"
                  stroke={token.position === state.selectedTokenPosition ? "#74e9b4" : "#416f5e"}
                  strokeWidth={token.position === state.selectedTokenPosition ? 3 : 1.5}
                  opacity={state.expandedStage && state.expandedStage !== stage.id && state.expandedStage !== nextStage.id ? 0.12 : 0.62}
                />
              );
            }))}
          </g>
          {TOY_STAGES.map((stage, stageIndex) => {
            const x = xForStage(stageIndex);
            const active = state.stage === stage.id || state.expandedStage === stage.id;
            const available = stage.available(result);
            const dimmed = state.expandedStage !== null && state.expandedStage !== stage.id;
            return (
              <g
                key={stage.id}
                className={styles.stageColumn}
                data-active={active ? "true" : "false"}
                data-dimmed={dimmed ? "true" : "false"}
                transform={`translate(${x} ${STAGE_TOP})`}
                role="button"
                tabIndex={0}
                aria-label={`${stage.title}${available ? "" : ", unavailable in Toy Math Lab"}`}
                onClick={() => selectStage(stage.id)}
                onKeyDown={(event) => onStageKeyDown(event, stage.id)}
              >
                <rect width={STAGE_WIDTH} height={stageHeight} rx="13" />
                <text className={styles.stageIndex} x="10" y="20">{String(stageIndex + 1).padStart(2, "0")}</text>
                <text className={styles.stageTitle} x={STAGE_WIDTH / 2} y="43" textAnchor="middle">{stage.shortLabel}</text>
                <line x1="9" x2={STAGE_WIDTH - 9} y1="55" y2="55" />
                {result.tokens.map((token) => {
                  const vector = stageVector(result, stage.id, token.position);
                  const selected = token.position === state.selectedTokenPosition;
                  return (
                    <g
                      key={token.position}
                      transform={`translate(9 ${67 + token.position * ROW_HEIGHT})`}
                      onClick={(event) => { event.stopPropagation(); dispatch({ type: "select-token", position: token.position }); }}
                      onMouseEnter={() => setHovered({ stage: stage.id, position: token.position })}
                      onMouseLeave={() => setHovered(null)}
                    >
                      <rect className={styles.tokenNode} data-selected={selected ? "true" : "false"} width={STAGE_WIDTH - 18} height="35" rx="8" />
                      <text className={styles.tokenLabel} x="7" y="14">{token.token}</text>
                      {stage.id === "tokens" ? <text className={styles.tokenValue} x="7" y="28">ID {token.token_id}</text> : null}
                      {vector ? vector.map((value, dimension) => (
                        <rect key={dimension} x={43 + dimension * 14} y="19" width="11" height="10" rx="2" fill={color(value)}>
                          <title>d{dimension}: {value.toFixed(4)}</title>
                        </rect>
                      )) : null}
                      {!available && token.position === 0 ? <text className={styles.unavailableSvg} x="7" y="28">trained only</text> : null}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </g>
      </svg>
      <nav className={styles.stageKeyboardNav} aria-label="Transformer stages">
        {TOY_STAGES.map((stage, index) => (
          <button type="button" key={stage.id} aria-current={state.stage === stage.id ? "step" : undefined} onClick={() => selectStage(stage.id)}>
            <span>{String(index + 1).padStart(2, "0")}</span>{stage.shortLabel}
          </button>
        ))}
      </nav>
    </div>
  );
}
