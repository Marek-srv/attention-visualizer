import { max, scaleLinear } from "d3";
import { useMemo } from "react";

import styles from "./visualizations.module.css";

type VectorRowProps = {
  label: string;
  values: readonly number[];
  selectedDimension?: number;
  onSelectDimension?: (dimension: number) => void;
  tone?: "input" | "query" | "key" | "value" | "output";
};

export default function VectorRow({
  label,
  values,
  selectedDimension,
  onSelectDimension,
  tone = "input",
}: VectorRowProps) {
  const color = useMemo(() => {
    const limit = max(values, (value) => Math.abs(value)) ?? 1;
    return scaleLinear<string>()
      .domain([-Math.max(limit, 1e-8), 0, Math.max(limit, 1e-8)])
      .range(["#d86f68", "#132c26", "#4bc98f"])
      .clamp(true);
  }, [values]);

  return (
    <div className={styles.vectorRow} data-tone={tone}>
      <strong>{label}</strong>
      <div className={styles.vectorCells} role="group" aria-label={`${label} vector`}>
        {values.map((value, dimension) => {
          const content = <><small>d{dimension}</small><span>{value.toFixed(4)}</span></>;
          const common = {
            className: styles.vectorCell,
            "data-selected": dimension === selectedDimension ? "true" : "false",
            "aria-label": `${label}, dimension ${dimension}: ${value.toFixed(4)}`,
            style: { backgroundColor: color(value) },
          } as const;
          return onSelectDimension ? (
            <button
              type="button"
              key={dimension}
              {...common}
              aria-pressed={dimension === selectedDimension}
              onClick={() => onSelectDimension(dimension)}
              onFocus={() => onSelectDimension(dimension)}
              onMouseEnter={() => onSelectDimension(dimension)}
            >
              {content}
            </button>
          ) : (
            <div key={dimension} {...common}>{content}</div>
          );
        })}
      </div>
    </div>
  );
}
