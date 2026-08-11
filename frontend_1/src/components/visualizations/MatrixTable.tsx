import styles from "./visualizations.module.css";

type MatrixTableProps = {
  title: string;
  values: ReadonlyArray<ReadonlyArray<number>>;
  highlightedColumn?: number;
  highlightedRow?: number;
};

export default function MatrixTable({ title, values, highlightedColumn, highlightedRow }: MatrixTableProps) {
  return (
    <div className={styles.matrixScroll}>
      <table className={styles.numericTable}>
        <caption>{title}</caption>
        <tbody>
          {values.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th scope="row">r{rowIndex}</th>
              {row.map((value, columnIndex) => (
                <td
                  key={columnIndex}
                  data-highlighted={rowIndex === highlightedRow || columnIndex === highlightedColumn ? "true" : "false"}
                >
                  {value.toFixed(4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

