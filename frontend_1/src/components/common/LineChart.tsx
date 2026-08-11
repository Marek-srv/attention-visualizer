import { curveMonotoneX, line, scaleLinear } from "d3";
import { gsap } from "gsap";
import { useEffect, useMemo, useRef } from "react";

import { usePrefersReducedMotion } from "../../hooks/useReducedMotion";

export type ChartPoint = {
  epoch: number;
  training: number;
  validation: number;
};

type LineChartProps = {
  points: readonly ChartPoint[];
  title: string;
  valueLabel: string;
  selectedEpoch?: number;
  onSelectEpoch?: (epoch: number) => void;
};

const WIDTH = 760;
const HEIGHT = 270;
const MARGIN = { top: 24, right: 24, bottom: 42, left: 62 };

export default function LineChart({
  points,
  title,
  valueLabel,
  selectedEpoch,
  onSelectEpoch,
}: LineChartProps) {
  const chartRef = useRef<SVGSVGElement | null>(null);
  const previousEpochs = useRef<Set<number>>(new Set());
  const reducedMotion = usePrefersReducedMotion();
  const geometry = useMemo(() => {
    if (points.length === 0) return null;
    const epochs = points.map((point) => point.epoch);
    const values = points.flatMap((point) => [point.training, point.validation]).filter(Number.isFinite);
    const minimumEpoch = Math.min(...epochs);
    const maximumEpoch = Math.max(...epochs);
    const maximumValue = Math.max(...values, 1e-6);
    const x = scaleLinear()
      .domain(minimumEpoch === maximumEpoch ? [minimumEpoch - 1, maximumEpoch + 1] : [minimumEpoch, maximumEpoch])
      .range([MARGIN.left, WIDTH - MARGIN.right]);
    const y = scaleLinear()
      .domain([0, maximumValue * 1.08])
      .nice()
      .range([HEIGHT - MARGIN.bottom, MARGIN.top]);
    const makeLine = (kind: "training" | "validation") => line<ChartPoint>()
      .defined((point) => Number.isFinite(point[kind]))
      .x((point) => x(point.epoch))
      .y((point) => y(point[kind]))
      .curve(curveMonotoneX)(points) ?? "";
    return { x, y, trainingPath: makeLine("training"), validationPath: makeLine("validation") };
  }, [points]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || points.length === 0) {
      previousEpochs.current = new Set(points.map((point) => point.epoch));
      return undefined;
    }

    const nextEpochs = new Set(points.map((point) => point.epoch));
    const enteringEpochs = points
      .map((point) => point.epoch)
      .filter((epoch) => !previousEpochs.current.has(epoch));
    previousEpochs.current = nextEpochs;

    if (reducedMotion) {
      gsap.killTweensOf(chart.querySelectorAll(".chart-line, .chart-point"));
      gsap.set(chart.querySelectorAll(".chart-line, .chart-point"), { clearProps: "all" });
      return undefined;
    }

    const context = gsap.context(() => {
      enteringEpochs.forEach((epoch) => {
        const point = chart.querySelector<SVGGElement>(`[data-chart-epoch="${epoch}"]`);
        if (!point) return;
        gsap.fromTo(
          point,
          { opacity: 0, y: 9, scale: 0.7, transformOrigin: "center" },
          { opacity: 1, y: 0, scale: 1, duration: 0.34, ease: "power2.out", clearProps: "transform,opacity" },
        );
      });

      if (enteringEpochs.length > 0) {
        chart.querySelectorAll<SVGPathElement>(".chart-line").forEach((path) => {
          if (typeof path.getTotalLength !== "function") return;
          const length = path.getTotalLength();
          const updateLength = Math.min(42, Math.max(8, length / Math.max(points.length, 1)));
          gsap.fromTo(
            path,
            { strokeDasharray: length, strokeDashoffset: updateLength },
            { strokeDashoffset: 0, duration: 0.38, ease: "power1.out", clearProps: "strokeDasharray,strokeDashoffset" },
          );
        });
      }
    }, chart);

    return () => context.revert();
  }, [points, reducedMotion]);

  if (!geometry) return <p className="empty-state">Metrics will appear after the first completed epoch.</p>;

  const ticks = geometry.y.ticks(4);
  return (
    <div className="line-chart-block">
      <svg ref={chartRef} className="line-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-labelledby={`${title}-chart-title`}>
        <title id={`${title}-chart-title`}>{title}: training and validation by epoch</title>
        <desc>Two aligned lines compare training and validation {valueLabel.toLowerCase()}.</desc>
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="chart-grid-line"
              x1={MARGIN.left}
              x2={WIDTH - MARGIN.right}
              y1={geometry.y(tick)}
              y2={geometry.y(tick)}
            />
            <text className="chart-axis-label" x={MARGIN.left - 10} y={geometry.y(tick) + 4} textAnchor="end">
              {tick.toFixed(tick < 1 ? 2 : 1)}
            </text>
          </g>
        ))}
        <line className="chart-axis" x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={HEIGHT - MARGIN.bottom} y2={HEIGHT - MARGIN.bottom} />
        <path className="chart-line chart-line--training" d={geometry.trainingPath} />
        <path className="chart-line chart-line--validation" d={geometry.validationPath} />
        {points.map((point) => (
          <g key={point.epoch} data-chart-epoch={point.epoch} className={selectedEpoch === point.epoch ? "chart-point chart-point--selected" : "chart-point"}>
            <circle cx={geometry.x(point.epoch)} cy={geometry.y(point.training)} r={selectedEpoch === point.epoch ? 6 : 4} />
            <circle cx={geometry.x(point.epoch)} cy={geometry.y(point.validation)} r={selectedEpoch === point.epoch ? 6 : 4} />
          </g>
        ))}
        <text className="chart-axis-title" x={WIDTH / 2} y={HEIGHT - 8} textAnchor="middle">Epoch</text>
        <text className="chart-axis-title" transform={`translate(16 ${HEIGHT / 2}) rotate(-90)`} textAnchor="middle">{valueLabel}</text>
      </svg>
      <div className="chart-legend" aria-hidden="true">
        <span><i className="legend-swatch legend-swatch--training" />Training</span>
        <span><i className="legend-swatch legend-swatch--validation" />Validation</span>
      </div>
      <div className="chart-epoch-controls" aria-label={`Select an epoch from ${title}`}>
        {points.map((point) => (
          <button
            type="button"
            key={point.epoch}
            aria-pressed={selectedEpoch === point.epoch}
            onClick={() => onSelectEpoch?.(point.epoch)}
          >
            Epoch {point.epoch}
          </button>
        ))}
      </div>
    </div>
  );
}
