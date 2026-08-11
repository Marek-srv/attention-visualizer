import styles from "./visualizations.module.css";

type EmbeddingAdditionTableProps = {
  tokenValues: readonly number[];
  positionValues: readonly number[];
  combinedValues: readonly number[];
  selectedDimension: number;
  onSelectDimension: (dimension: number) => void;
};

export default function EmbeddingAdditionTable({
  tokenValues,
  positionValues,
  combinedValues,
  selectedDimension,
  onSelectDimension,
}: EmbeddingAdditionTableProps) {
  return (
    <div className={styles.matrixScroll}>
      <table className={styles.embeddingTable}>
        <caption>Exact embedding addition by dimension</caption>
        <thead>
          <tr><th scope="col">Dimension</th><th scope="col">Token embedding</th><th scope="col">Position embedding</th><th scope="col">Combined X</th></tr>
        </thead>
        <tbody>
          {combinedValues.map((combined, dimension) => (
            <tr
              key={dimension}
              data-selected={dimension === selectedDimension ? "true" : "false"}
              onMouseEnter={() => onSelectDimension(dimension)}
              onFocusCapture={() => onSelectDimension(dimension)}
            >
              <th scope="row"><button type="button" aria-pressed={dimension === selectedDimension} onClick={() => onSelectDimension(dimension)}>d{dimension}</button></th>
              <td>{(tokenValues[dimension] ?? 0).toFixed(4)}</td>
              <td>+ {(positionValues[dimension] ?? 0).toFixed(4)}</td>
              <td>= {combined.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
