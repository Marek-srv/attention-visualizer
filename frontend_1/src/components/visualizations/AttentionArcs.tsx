import { scaleLinear } from "d3";
import { useMemo } from "react";

import styles from "./visualizations.module.css";

type AttentionArcsProps = {
  tokens: readonly string[];
  queryPosition: number;
  weights: readonly number[];
  allowed: readonly boolean[];
  onSelectKey?: (position: number) => void;
  selectedKey?: number;
};

export default function AttentionArcs({
  tokens,
  queryPosition,
  weights,
  allowed,
  onSelectKey,
  selectedKey,
}: AttentionArcsProps) {
  const width = 760;
  const height = 220;
  const tokenX = (index: number): number => 55 + (index * (width - 110)) / Math.max(tokens.length - 1, 1);
  const stroke = useMemo(() => scaleLinear().domain([0, 1]).range([1, 13]).clamp(true), []);
  const opacity = useMemo(() => scaleLinear().domain([0, 1]).range([0.18, 0.95]).clamp(true), []);
  const sourceX = tokenX(queryPosition);

  return (
    <div className={styles.arcRegion}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="attention-arcs-title attention-arcs-description">
        <title id="attention-arcs-title">Attention from {tokens[queryPosition] ?? `token ${queryPosition}`}</title>
        <desc id="attention-arcs-description">Arc width and opacity encode attention probability. Future tokens are not connected.</desc>
        {tokens.map((_, keyPosition) => {
          if (!allowed[keyPosition]) return null;
          const keyX = tokenX(keyPosition);
          const probability = weights[keyPosition] ?? 0;
          const rise = 38 + Math.abs(keyX - sourceX) * 0.22;
          return (
            <path
              key={`arc-${keyPosition}`}
              d={`M ${sourceX} 174 Q ${(sourceX + keyX) / 2} ${174 - rise} ${keyX} 174`}
              fill="none"
              stroke={keyPosition === selectedKey ? "#f3c66f" : "#6de2ac"}
              strokeWidth={stroke(probability)}
              opacity={opacity(probability)}
            />
          );
        })}
        {tokens.map((token, position) => (
          <g key={position} transform={`translate(${tokenX(position)}, 184)`}>
            <circle r="22" fill={position === queryPosition ? "#245945" : "#102820"} stroke={position === selectedKey ? "#f3c66f" : "#5a9d7f"} />
            <text textAnchor="middle" dominantBaseline="middle" fill="#eef8f3" fontSize="12">{token}</text>
          </g>
        ))}
      </svg>
      <div className={styles.arcControls} role="group" aria-label="Select attention key token">
        {tokens.map((token, position) => {
          const content = <>{token}: {allowed[position] ? `${((weights[position] ?? 0) * 100).toFixed(1)}%` : "Masked"}</>;
          return onSelectKey ? (
            <button type="button" key={position} disabled={!allowed[position]} aria-pressed={position === selectedKey} onClick={() => onSelectKey(position)}>{content}</button>
          ) : (
            <span key={position} data-masked={!allowed[position] ? "true" : "false"}>{content}</span>
          );
        })}
      </div>
      <p className={styles.legend}>Only here does line width encode a value: the selected query token&apos;s attention probability.</p>
    </div>
  );
}
