import { RotateCcw, Settings2, Sparkles } from "lucide-react";
import { useState, type ReactNode } from "react";

import type {
  NumericMatrix,
  ToyInspectResponse,
} from "../../types/api";
import { cloneFeedForward, feedForwardParametersFromResponse } from "./toyAdapter";
import type { ToyControlValues } from "./toyControlRequest";
import styles from "./ToyLabPage.module.css";

type ControlTab = "query" | "key" | "value" | "output" | "norm-1" | "ffn" | "norm-2";

type AdvancedControlsProps = {
  result: ToyInspectResponse;
  defaults: ToyInspectResponse;
  busy: boolean;
  onRecalculate: (values: ToyControlValues) => void;
  onReset: () => void;
};

function valuesFromResponse(result: ToyInspectResponse): ToyControlValues {
  return {
    weights: {
      query: result.weights.query.map((row) => [...row]),
      key: result.weights.key.map((row) => [...row]),
      value: result.weights.value.map((row) => [...row]),
      output: result.weights.output?.map((row) => [...row]) ?? result.multi_head_attention.output_weight_matrix.map((row) => [...row]),
    },
    normalization: {
      gamma: [...result.attention_sublayer.gamma],
      beta: [...result.attention_sublayer.beta],
      epsilon: result.attention_sublayer.epsilon,
    },
    feedForward: feedForwardParametersFromResponse(result),
  };
}

function MatrixEditor({ label, matrix, onChange }: { label: string; matrix: NumericMatrix; onChange: (row: number, column: number, value: number) => void }) {
  return (
    <div className={styles.editorScroll}>
      <table className={styles.editorTable}>
        <caption>{label}</caption>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <th scope="row">r{rowIndex}</th>
              {row.map((value, columnIndex) => (
                <td key={columnIndex}>
                  <label>
                    <span>row {rowIndex}, column {columnIndex}</span>
                    <input type="number" step="0.01" value={value} onChange={(event) => {
                      const next = event.currentTarget.valueAsNumber;
                      if (Number.isFinite(next)) onChange(rowIndex, columnIndex, next);
                    }} />
                  </label>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VectorEditor({ label, values, symbol, onChange }: { label: string; values: readonly number[]; symbol: string; onChange: (index: number, value: number) => void }) {
  return (
    <fieldset className={styles.vectorEditor}>
      <legend>{label}</legend>
      {values.map((value, index) => (
        <label key={index}>
          <span>{symbol}{index}</span>
          <input type="number" step="0.01" value={value} onChange={(event) => {
            const next = event.currentTarget.valueAsNumber;
            if (Number.isFinite(next)) onChange(index, next);
          }} />
        </label>
      ))}
    </fieldset>
  );
}

function Panel({ children }: { children: ReactNode }) {
  return <div className={styles.controlPanel}>{children}</div>;
}

export default function AdvancedControls({ result, defaults, busy, onRecalculate, onReset }: AdvancedControlsProps) {
  const [tab, setTab] = useState<ControlTab>("query");
  const [values, setValues] = useState<ToyControlValues>(() => valuesFromResponse(result));

  const updateWeight = (kind: "query" | "key" | "value" | "output", row: number, column: number, value: number): void => {
    setValues((current) => {
      const source = current.weights[kind] ?? [];
      const matrix = source.map((line) => [...line]);
      if (matrix[row]) matrix[row][column] = value;
      return { ...current, weights: { ...current.weights, [kind]: matrix } };
    });
  };

  const updateNormalization = (kind: "gamma" | "beta", index: number, value: number): void => {
    setValues((current) => ({
      ...current,
      normalization: { ...current.normalization, [kind]: current.normalization[kind].map((item, itemIndex) => itemIndex === index ? value : item) },
    }));
  };

  const updateFeedForwardMatrix = (kind: "input_weights" | "output_weights", row: number, column: number, value: number): void => {
    setValues((current) => {
      const next = cloneFeedForward(current.feedForward);
      if (next[kind][row]) next[kind][row][column] = value;
      return { ...current, feedForward: next };
    });
  };

  const updateFeedForwardVector = (kind: "input_bias" | "output_bias", index: number, value: number): void => {
    setValues((current) => {
      const next = cloneFeedForward(current.feedForward);
      next[kind][index] = value;
      return { ...current, feedForward: next };
    });
  };

  const updateFeedForwardNormalization = (kind: "gamma" | "beta", index: number, value: number): void => {
    setValues((current) => {
      const next = cloneFeedForward(current.feedForward);
      next.normalization[kind][index] = value;
      return { ...current, feedForward: next };
    });
  };

  const resetLocally = (): void => setValues(valuesFromResponse(defaults));

  return (
    <details className={styles.advanced}>
      <summary><Settings2 aria-hidden="true" size={17} /><span><strong>Advanced Controls</strong><small>Edit Q/K/V, WO, both normalizations, and FFN parameters.</small></span></summary>
      <div className={styles.advancedBody}>
        <p className={styles.controlWarning}>Every change is sent to the existing `/api/inspect` endpoint. The connected visualization updates only after a successful recalculation.</p>
        <div className={styles.controlTabs} role="tablist" aria-label="Parameter group">
          {([
            ["query", "WQ"], ["key", "WK"], ["value", "WV"], ["output", "WO"], ["norm-1", "LayerNorm 1"], ["ffn", "FFN"], ["norm-2", "LayerNorm 2"],
          ] as const).map(([id, label]) => <button type="button" role="tab" aria-selected={tab === id} key={id} onClick={() => setTab(id)}>{label}</button>)}
        </div>
        {tab === "query" || tab === "key" || tab === "value" || tab === "output" ? (
          <Panel><MatrixEditor label={`${tab === "output" ? "WO" : `W${tab[0]?.toUpperCase()}`} matrix`} matrix={values.weights[tab] ?? []} onChange={(row, column, value) => updateWeight(tab, row, column, value)} /></Panel>
        ) : null}
        {tab === "norm-1" ? (
          <Panel>
            <VectorEditor label="Gamma — scale" symbol="γ" values={values.normalization.gamma} onChange={(index, value) => updateNormalization("gamma", index, value)} />
            <VectorEditor label="Beta — shift" symbol="β" values={values.normalization.beta} onChange={(index, value) => updateNormalization("beta", index, value)} />
            <label className={styles.epsilonField}><span>Epsilon</span><input type="number" step="0.00001" min="0.0000001" value={values.normalization.epsilon} onChange={(event) => {
              const epsilon = event.currentTarget.valueAsNumber;
              if (Number.isFinite(epsilon) && epsilon > 0) setValues((current) => ({ ...current, normalization: { ...current.normalization, epsilon } }));
            }} /></label>
          </Panel>
        ) : null}
        {tab === "ffn" ? (
          <Panel>
            <MatrixEditor label="W1 · 4 × 8" matrix={values.feedForward.input_weights} onChange={(row, column, value) => updateFeedForwardMatrix("input_weights", row, column, value)} />
            <VectorEditor label="b1 · 8" symbol="b" values={values.feedForward.input_bias} onChange={(index, value) => updateFeedForwardVector("input_bias", index, value)} />
            <MatrixEditor label="W2 · 8 × 4" matrix={values.feedForward.output_weights} onChange={(row, column, value) => updateFeedForwardMatrix("output_weights", row, column, value)} />
            <VectorEditor label="b2 · 4" symbol="b" values={values.feedForward.output_bias} onChange={(index, value) => updateFeedForwardVector("output_bias", index, value)} />
          </Panel>
        ) : null}
        {tab === "norm-2" ? (
          <Panel>
            <VectorEditor label="Gamma — scale" symbol="γ" values={values.feedForward.normalization.gamma} onChange={(index, value) => updateFeedForwardNormalization("gamma", index, value)} />
            <VectorEditor label="Beta — shift" symbol="β" values={values.feedForward.normalization.beta} onChange={(index, value) => updateFeedForwardNormalization("beta", index, value)} />
            <label className={styles.epsilonField}><span>Epsilon</span><input type="number" step="0.00001" min="0.0000001" value={values.feedForward.normalization.epsilon} onChange={(event) => {
              const epsilon = event.currentTarget.valueAsNumber;
              if (Number.isFinite(epsilon) && epsilon > 0) setValues((current) => ({ ...current, feedForward: { ...current.feedForward, normalization: { ...current.feedForward.normalization, epsilon } } }));
            }} /></label>
          </Panel>
        ) : null}
        <div className={styles.controlActions}>
          <button type="button" disabled={busy} onClick={() => onRecalculate(values)}><Sparkles aria-hidden="true" size={15} /> Recalculate from backend</button>
          <button type="button" disabled={busy} onClick={() => { resetLocally(); onReset(); }}><RotateCcw aria-hidden="true" size={15} /> Reset all defaults</button>
        </div>
      </div>
    </details>
  );
}
