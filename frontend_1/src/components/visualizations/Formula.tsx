import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

import styles from "./visualizations.module.css";

type FormulaProps = {
  expression: string;
  label?: string;
  block?: boolean;
};

export default function Formula({ expression, label = "Formula", block = true }: FormulaProps) {
  const markup = useMemo(
    () => katex.renderToString(expression, { displayMode: block, throwOnError: false, strict: false }),
    [block, expression],
  );

  return (
    <div
      className={styles.formula}
      aria-label={`${label}: ${expression}`}
      dangerouslySetInnerHTML={{ __html: markup }}
    />
  );
}

