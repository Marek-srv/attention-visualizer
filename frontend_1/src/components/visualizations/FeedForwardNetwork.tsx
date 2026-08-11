import { max, scaleLinear } from "d3";
import { useMemo } from "react";

import styles from "./visualizations.module.css";

type FeedForwardNetworkProps = {
  input: readonly number[];
  hiddenPreActivation: readonly number[];
  hiddenActivation: readonly number[];
  output: readonly number[];
  activationName: string;
  selectedNeuron: number;
  onSelectNeuron: (neuron: number) => void;
};

export default function FeedForwardNetwork({
  input,
  hiddenPreActivation,
  hiddenActivation,
  output,
  activationName,
  selectedNeuron,
  onSelectNeuron,
}: FeedForwardNetworkProps) {
  const width = 760;
  const height = 270;
  const nodeY = (index: number, count: number): number => 32 + index * ((height - 64) / Math.max(count - 1, 1));
  const color = useMemo(() => {
    const values = [...input, ...hiddenActivation, ...output];
    const extent = Math.max(max(values, (value) => Math.abs(value)) ?? 1, 1e-8);
    return scaleLinear<string>().domain([-extent, 0, extent]).range(["#d86f68", "#17352d", "#58d39b"]).clamp(true);
  }, [hiddenActivation, input, output]);

  return (
    <section className={styles.ffnNetwork} aria-labelledby="ffn-network-title">
      <div className={styles.networkHeading}>
        <div><span>Accessible network flow</span><h3 id="ffn-network-title">4 inputs → {hiddenActivation.length} hidden neurons → 4 outputs</h3></div>
        <p>Connector width shows the route only. Node labels show backend-derived values and activation state.</p>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="ffn-svg-title ffn-svg-description">
        <title id="ffn-svg-title">Feed-forward network flow</title>
        <desc id="ffn-svg-description">Four input features connect to {hiddenActivation.length} hidden neurons, then return to four model features. Select a neuron using the controls below.</desc>
        <g className={styles.networkLinks} aria-hidden="true">
          {input.flatMap((_, inputIndex) => hiddenActivation.map((__, hiddenIndex) => (
            <line key={`in-${inputIndex}-${hiddenIndex}`} x1="92" y1={nodeY(inputIndex, input.length)} x2="366" y2={nodeY(hiddenIndex, hiddenActivation.length)} />
          )))}
          {hiddenActivation.flatMap((_, hiddenIndex) => output.map((__, outputIndex) => (
            <line key={`out-${hiddenIndex}-${outputIndex}`} x1="394" y1={nodeY(hiddenIndex, hiddenActivation.length)} x2="668" y2={nodeY(outputIndex, output.length)} />
          )))}
        </g>
        <text className={styles.networkLabel} x="64" y="17" textAnchor="middle">Input · 4</text>
        <text className={styles.networkLabel} x="380" y="17" textAnchor="middle">{activationName} hidden · {hiddenActivation.length}</text>
        <text className={styles.networkLabel} x="696" y="17" textAnchor="middle">Output · {output.length}</text>
        {input.map((value, index) => (
          <g key={index} transform={`translate(64 ${nodeY(index, input.length)})`}>
            <circle r="25" fill={color(value)} stroke="#64b791" />
            <text textAnchor="middle" dominantBaseline="middle">d{index}</text>
            <title>Input d{index}: {value.toFixed(4)}</title>
          </g>
        ))}
        {hiddenActivation.map((value, index) => {
          const active = value > 0;
          return (
            <g key={index} transform={`translate(380 ${nodeY(index, hiddenActivation.length)})`} data-selected={index === selectedNeuron ? "true" : "false"}>
              <circle r="19" fill={color(value)} stroke={index === selectedNeuron ? "#f3c66f" : "#7c6dae"} strokeWidth={index === selectedNeuron ? 3 : 1.3} />
              <text textAnchor="middle" dominantBaseline="middle">h{index}</text>
              <text className={styles.networkState} x="25" y="4">{active ? "Active" : "Inactive"}</text>
              <title>Hidden neuron {index}: pre-activation {(hiddenPreActivation[index] ?? 0).toFixed(4)}, {activationName} output {value.toFixed(4)}, {active ? "Active" : "Inactive"}</title>
            </g>
          );
        })}
        {output.map((value, index) => (
          <g key={index} transform={`translate(696 ${nodeY(index, output.length)})`}>
            <circle r="25" fill={color(value)} stroke="#64b791" />
            <text textAnchor="middle" dominantBaseline="middle">d{index}</text>
            <title>Output d{index}: {value.toFixed(4)}</title>
          </g>
        ))}
      </svg>
      <div className={styles.neuronControls} role="group" aria-label="Select a hidden neuron for its exact weighted sum">
        {hiddenActivation.map((value, neuron) => (
          <button type="button" key={neuron} aria-pressed={neuron === selectedNeuron} onClick={() => onSelectNeuron(neuron)}>
            <strong>h{neuron}</strong>
            <span>pre {(hiddenPreActivation[neuron] ?? 0).toFixed(4)}</span>
            <span>{activationName} {value.toFixed(4)}</span>
            <em>{value > 0 ? "Active" : "Inactive"}</em>
          </button>
        ))}
      </div>
    </section>
  );
}

