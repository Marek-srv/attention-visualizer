import { max, scaleLinear } from "d3";
import { useMemo } from "react";

import styles from "./visualizations.module.css";

export type MatrixSelection = { row: number; column: number };

type MatrixHeatmapProps = {
  title: string;
  values: ReadonlyArray<ReadonlyArray<number | null>>;
  rowLabels?: readonly string[];
  columnLabels?: readonly string[];
  allowedMask?: ReadonlyArray<ReadonlyArray<boolean>>;
  selected?: MatrixSelection;
  onSelect?: (selection: MatrixSelection) => void;
  rowSums?: readonly number[];
  compact?: boolean;
};

export default function MatrixHeatmap({
  title,
  values,
  rowLabels,
  columnLabels,
  allowedMask,
  selected,
  onSelect,
  rowSums,
  compact = false,
}: MatrixHeatmapProps) {
  const color = useMemo(() => {
    const magnitude = max(
      values.flatMap((row) => row).filter((value): value is number => value !== null),
      (value) => Math.abs(value),
    ) ?? 1;
    const extent = Math.max(magnitude, 1e-8);
    return scaleLinear<string>().domain([-extent, 0, extent]).range(["#c9625f", "#122821", "#49c98e"]).clamp(true);
  }, [values]);

  const columnCount = max(values, (row) => row.length) ?? 0;

  return (
    <div className={styles.matrixRegion} data-compact={compact ? "true" : "false"}>
      <div className={styles.matrixScroll}>
        <table className={styles.matrix}>
          <caption>{title}</caption>
          <thead>
            <tr>
              <th scope="col">Query</th>
              {Array.from({ length: columnCount }, (_, column) => (
                <th scope="col" key={column}>{columnLabels?.[column] ?? `k${column}`}</th>
              ))}
              {rowSums ? <th scope="col">Row sum</th> : null}
            </tr>
          </thead>
          <tbody>
            {values.map((row, rowIndex) => (
              <tr key={rowIndex} data-selected={selected?.row === rowIndex ? "true" : "false"}>
                <th scope="row">{rowLabels?.[rowIndex] ?? `q${rowIndex}`}</th>
                {row.map((value, columnIndex) => {
                  const masked = value === null || allowedMask?.[rowIndex]?.[columnIndex] === false;
                  const active = selected?.row === rowIndex && selected.column === columnIndex;
                  const label = masked
                    ? `${rowLabels?.[rowIndex] ?? `query ${rowIndex}`} to ${columnLabels?.[columnIndex] ?? `key ${columnIndex}`}: Masked, weight zero`
                    : `${rowLabels?.[rowIndex] ?? `query ${rowIndex}`} to ${columnLabels?.[columnIndex] ?? `key ${columnIndex}`}: ${(value ?? 0).toFixed(4)}`;
                  const content = masked ? <><span>Masked</span><small>0.0000</small></> : (value ?? 0).toFixed(4);
                  const common = {
                    className: styles.matrixCell,
                    "data-masked": masked ? "true" : "false",
                    "data-selected": active ? "true" : "false",
                    "aria-label": label,
                    style: masked ? undefined : { backgroundColor: color(value ?? 0) },
                  } as const;
                  return (
                    <td key={columnIndex}>
                      {onSelect ? (
                        <button type="button" {...common} aria-pressed={active} onClick={() => onSelect({ row: rowIndex, column: columnIndex })}>{content}</button>
                      ) : (
                        <div {...common}>{content}</div>
                      )}
                    </td>
                  );
                })}
                {rowSums ? (
                  <td className={styles.rowSum} data-valid={Math.abs((rowSums[rowIndex] ?? 0) - 1) <= 0.001 ? "true" : "false"}>
                    {(rowSums[rowIndex] ?? 0).toFixed(4)}
                    <small>{Math.abs((rowSums[rowIndex] ?? 0) - 1) <= 0.001 ? "≈ 1" : "check"}</small>
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={styles.legend}>Positive values use emerald, negative values use coral, and masked future positions use a labelled hatch.</p>
    </div>
  );
}
