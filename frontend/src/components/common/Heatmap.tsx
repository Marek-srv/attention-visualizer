type HeatmapProps = {
  title?: string;
  values: Array<Array<number | null>>;
  rowLabels?: string[];
  columnLabels?: string[];
  format?: (value: number) => string;
  selectedRow?: number;
  selectedCell?: [number, number];
  onCellSelect?: (row: number, column: number) => void;
  emptyLabel?: string;
};

function heatColor(value: number, maximum: number): string {
  const strength = maximum > 0 ? Math.min(Math.abs(value) / maximum, 1) : 0;
  return value >= 0
    ? `rgba(77, 205, 139, ${0.11 + strength * 0.72})`
    : `rgba(244, 119, 119, ${0.11 + strength * 0.68})`;
}

export default function Heatmap({
  title,
  values,
  rowLabels = [],
  columnLabels = [],
  format = (value) => value.toFixed(4),
  selectedRow,
  selectedCell,
  onCellSelect,
  emptyLabel = "No tensor values are available yet.",
}: HeatmapProps) {
  const columnCount = Math.max(0, ...values.map((row) => row.length));
  const finiteValues = values.flatMap((row) => row).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const maximum = Math.max(0, ...finiteValues.map(Math.abs));

  if (!values.length || !columnCount) return <div className="empty-state compact" role="status">{emptyLabel}</div>;

  return (
    <div className="matrix-scroll">
      <table className="tensor-table">
        {title && <caption>{title}</caption>}
        <thead><tr><th scope="col">Token</th>{Array.from({ length: columnCount }, (_, column) => <th scope="col" key={column}>{columnLabels[column] ?? `d${column}`}</th>)}</tr></thead>
        <tbody>{values.map((row, rowIndex) => <tr className={selectedRow === rowIndex ? "selected-row" : ""} key={rowIndex}><th scope="row">{rowLabels[rowIndex] ?? `t${rowIndex}`}</th>{Array.from({ length: columnCount }, (_, columnIndex) => {
          const value = row[columnIndex];
          const selected = selectedCell?.[0] === rowIndex && selectedCell[1] === columnIndex;
          const contents = value === null || value === undefined ? "Masked" : format(value);
          const className = `${value === null || value === undefined ? "tensor-cell masked" : "tensor-cell"}${selected ? " selected" : ""}`;
          const style = typeof value === "number" ? { background: heatColor(value, maximum) } : undefined;
          return <td key={columnIndex}>{onCellSelect ? <button type="button" className={className} style={style} onClick={() => onCellSelect(rowIndex, columnIndex)} aria-label={`${rowLabels[rowIndex] ?? `token ${rowIndex}`}, ${columnLabels[columnIndex] ?? `dimension ${columnIndex}`}: ${contents}`}>{contents}</button> : <span className={className} style={style}>{contents}</span>}</td>;
        })}</tr>)}</tbody>
      </table>
    </div>
  );
}

