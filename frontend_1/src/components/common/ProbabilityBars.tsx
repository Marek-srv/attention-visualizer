import { useMemo } from "react";

export type ProbabilityDatum = {
  token: string;
  token_id: number;
  logit: number;
  probability: number;
};

type ProbabilityBarsProps = {
  predictions: readonly ProbabilityDatum[];
  selectedTokenId?: number;
  onSelect?: (prediction: ProbabilityDatum) => void;
  showTechnicalValues?: boolean;
  label?: string;
};

function visibleToken(token: string): string {
  if (token.length === 0) return "<empty>";
  return token.replaceAll(" ", "␠").replaceAll("\n", "↵");
}

export default function ProbabilityBars({
  predictions,
  selectedTokenId,
  onSelect,
  showTechnicalValues = false,
  label = "Model next-token probabilities",
}: ProbabilityBarsProps) {
  const ordered = useMemo(
    () => [...predictions].sort((left, right) => right.probability - left.probability),
    [predictions],
  );

  if (ordered.length === 0) {
    return <p className="empty-state">No probability candidates are available.</p>;
  }

  return (
    <ol className="probability-bars" aria-label={label}>
      {ordered.map((prediction, index) => {
        const percent = Math.max(0, Math.min(100, prediction.probability * 100));
        const content = (
          <>
            <span className="probability-rank" aria-hidden="true">{index + 1}</span>
            <code className="probability-token">{visibleToken(prediction.token)}</code>
            <span className="probability-track" aria-hidden="true">
              <span style={{ width: `${percent}%` }} />
            </span>
            <strong>{percent.toFixed(2)}%</strong>
            {showTechnicalValues ? (
              <small>
                ID {prediction.token_id} · logit {prediction.logit.toFixed(4)}
              </small>
            ) : null}
          </>
        );

        return (
          <li key={`${prediction.token_id}-${index}`}>
            {onSelect ? (
              <button
                type="button"
                className="probability-row"
                aria-pressed={selectedTokenId === prediction.token_id}
                onClick={() => onSelect(prediction)}
              >
                {content}
              </button>
            ) : (
              <div className="probability-row">{content}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
