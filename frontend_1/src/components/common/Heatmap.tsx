import { scaleLinear } from "d3";
import { useMemo } from "react";

export type HeatmapSelection = { row: number; column: number };

type HeatmapProps = {
  values: ReadonlyArray<ReadonlyArray<number | null>>;
  title: string;
  rowLabels?: readonly string[];
  columnLabels?: readonly string[];
  mask?: ReadonlyArray<ReadonlyArray<boolean>>;
  selected?: HeatmapSelection;
  onSelect?: (selection: HeatmapSelection) => void;
  formatValue?: (value: number) => string;
};

export default function Heatmap({
  values,
  title,
  rowLabels,
  columnLabels,
  mask,
  selected,
  onSelect,
  formatValue = (value) => value.toFixed(4),
}: HeatmapProps) {
  const color = useMemo(() => {
    const finite = values.flatMap((row) => row).filter((value): value is number => value !== null && Number.isFinite(value));
    const maximum = Math.max(...finite.map((value) => Math.abs(value)), 1e-8);
    return scaleLinear<string>()
      .domain([-maximum, 0, maximum])
      .range(["#a85550", "#12231f", "#43bd87"])
      .clamp(true);
  }, [values]);

  if (values.length === 0) return <p className="empty-state">No matrix values are available.</p>;
  const columns = Math.max(...values.map((row) => row.length), 0);

  return (
    <div className="heatmap-region" role="group" aria-label={title}>
      <div className="matrix-scroll">
        <table className="heatmap-table">
          <caption>{title}</caption>
          <thead>
            <tr>
              <th scope="col">Row</th>
              {Array.from({ length: columns }, (_, column) => (
                <th scope="col" key={column}>{columnLabels?.[column] ?? `d${column}`}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {values.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <th scope="row">{rowLabels?.[rowIndex] ?? rowIndex}</th>
                {row.map((value, columnIndex) => {
                  const isMasked = mask?.[rowIndex]?.[columnIndex] === true || value === null;
                  const isSelected = selected?.row === rowIndex && selected.column === columnIndex;
                  const label = isMasked
                    ? `${rowLabels?.[rowIndex] ?? `row ${rowIndex}`} to ${columnLabels?.[columnIndex] ?? `column ${columnIndex}`}: Masked, weight zero`
                    : `${rowLabels?.[rowIndex] ?? `row ${rowIndex}`}, ${columnLabels?.[columnIndex] ?? `column ${columnIndex}`}: ${formatValue(value ?? 0)}`;
                  const className = `heatmap-cell${isMasked ? " heatmap-cell--masked" : ""}${isSelected ? " heatmap-cell--selected" : ""}${onSelect ? "" : " heatmap-cell--static"}`;
                  const style = isMasked ? { display: "grid" } : { backgroundColor: color(value ?? 0), display: "grid" };
                  const content = isMasked ? <><span>Masked</span><small>0.0000</small></> : formatValue(value ?? 0);
                  return (
                    <td key={columnIndex}>
                      {onSelect ? (
                        <button
                          type="button"
                          className={className}
                          data-masked={isMasked ? "true" : "false"}
                          disabled={isMasked}
                          aria-label={label}
                          aria-pressed={isSelected}
                          style={style}
                          onClick={() => onSelect({ row: rowIndex, column: columnIndex })}
                        >
                          {content}
                        </button>
                      ) : (
                        <span
                          className={className}
                          data-masked={isMasked ? "true" : "false"}
                          aria-label={label}
                          style={style}
                        >
                          {content}
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="visualization-key">Emerald: positive · coral: negative · values remain labelled.</p>
    </div>
  );
}
