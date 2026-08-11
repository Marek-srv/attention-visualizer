import { useId } from "react";

export type ChartSeries = {
  label: string;
  values: number[];
  color: string;
};

type LineChartProps = {
  title: string;
  xLabels: Array<string | number>;
  series: ChartSeries[];
  valueFormatter?: (value: number) => string;
};

export default function LineChart({ title, xLabels, series, valueFormatter = (value) => value.toFixed(3) }: LineChartProps) {
  const titleId = useId();
  const descriptionId = useId();
  const width = 720;
  const height = 250;
  const padding = { top: 24, right: 24, bottom: 42, left: 58 };
  const values = series.flatMap((item) => item.values).filter(Number.isFinite);
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const range = Math.max(maximum - minimum, 0.000001);
  const maxPoints = Math.max(1, ...series.map((item) => item.values.length));
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const x = (index: number) => padding.left + (maxPoints === 1 ? plotWidth / 2 : index * plotWidth / (maxPoints - 1));
  const y = (value: number) => padding.top + (maximum - value) * plotHeight / range;
  const gridValues = Array.from({ length: 4 }, (_, index) => minimum + range * index / 3);

  if (!values.length) {
    return <div className="chart-empty" role="status">The chart will appear after the first completed epoch.</div>;
  }

  return (
    <div className="chart-wrap">
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
        <title id={titleId}>{title}</title>
        <desc id={descriptionId}>{series.map((item) => `${item.label}: ${item.values.map(valueFormatter).join(", ")}`).join(". ")}</desc>
        {gridValues.map((value) => (
          <g key={value}>
            <line x1={padding.left} x2={width - padding.right} y1={y(value)} y2={y(value)} className="chart-grid-line" />
            <text x={padding.left - 10} y={y(value) + 4} textAnchor="end" className="chart-axis-label">{valueFormatter(value)}</text>
          </g>
        ))}
        {series.map((item) => {
          const points = item.values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
          return (
            <g key={item.label}>
              <polyline points={points} fill="none" stroke={item.color} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" />
              {item.values.map((value, index) => <circle key={index} cx={x(index)} cy={y(value)} r="4" fill={item.color}><title>{`${item.label}, epoch ${xLabels[index] ?? index + 1}: ${valueFormatter(value)}`}</title></circle>)}
            </g>
          );
        })}
        {xLabels.length > 0 && <>
          <text x={padding.left} y={height - 13} className="chart-axis-label">Epoch {xLabels[0]}</text>
          <text x={width - padding.right} y={height - 13} textAnchor="end" className="chart-axis-label">Epoch {xLabels[xLabels.length - 1]}</text>
        </>}
      </svg>
      <div className="chart-legend" aria-hidden="true">{series.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>
      <details className="chart-data">
        <summary>Read chart values</summary>
        <div className="matrix-scroll"><table><thead><tr><th>Epoch</th>{series.map((item) => <th key={item.label}>{item.label}</th>)}</tr></thead><tbody>{xLabels.map((label, index) => <tr key={index}><th>{label}</th>{series.map((item) => <td key={item.label}>{item.values[index] === undefined ? "—" : valueFormatter(item.values[index])}</td>)}</tr>)}</tbody></table></div>
      </details>
    </div>
  );
}
