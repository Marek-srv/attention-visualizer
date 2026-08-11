import type { PredictionItem } from "../../types/api";

type ProbabilityBarsProps = {
  predictions: PredictionItem[];
  emptyLabel?: string;
};

export default function ProbabilityBars({ predictions, emptyLabel = "Run a prediction to see model probabilities." }: ProbabilityBarsProps) {
  if (!predictions.length) return <div className="empty-state compact" role="status">{emptyLabel}</div>;
  const maximum = Math.max(...predictions.map((item) => item.probability), 0.000001);
  return <div className="probability-list">{predictions.map((item) => <div className="probability-row" key={`${item.token_id}-${item.token}`}><div className="probability-token"><strong>{item.token === " " ? "␠" : item.token}</strong><span>ID {item.token_id}</span></div><div className="probability-track" aria-hidden="true"><span style={{ width: `${Math.max(1.5, item.probability / maximum * 100)}%` }} /></div><div className="probability-values"><span>{(item.probability * 100).toFixed(2)}%</span><small>logit {item.logit.toFixed(4)}</small></div></div>)}</div>;
}

