import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import MatrixHeatmap, { type MatrixSelection } from "./MatrixHeatmap";

function SelectableHeatmap() {
  const [selection, setSelection] = useState<MatrixSelection>({ row: 0, column: 0 });
  return (
    <>
      <MatrixHeatmap
        title="Causal attention"
        values={[[1, 0], [0.4, 0.6]]}
        allowedMask={[[true, false], [true, true]]}
        rowLabels={["I", "love"]}
        columnLabels={["I", "love"]}
        rowSums={[1, 1]}
        selected={selection}
        onSelect={setSelection}
      />
      <output aria-label="Selected calculation">cell {selection.row},{selection.column}</output>
    </>
  );
}

describe("MatrixHeatmap", () => {
  it("renders future causal positions as labelled masks and shows normalized row sums", () => {
    render(<SelectableHeatmap />);

    expect(screen.getByRole("button", { name: "I to love: Masked, weight zero" })).toHaveAttribute("data-masked", "true");
    expect(screen.getAllByText("≈ 1")).toHaveLength(2);
  });

  it("updates the calculation inspector when a heatmap cell is selected", async () => {
    const user = userEvent.setup();
    render(<SelectableHeatmap />);

    await user.click(screen.getByRole("button", { name: "love to I: 0.4000" }));

    expect(screen.getByRole("status", { name: "Selected calculation" })).toHaveTextContent("cell 1,0");
  });

  it("does not create focusable no-op cells when no selection callback exists", () => {
    render(<MatrixHeatmap title="Static values" values={[[0.25, -0.5]]} />);

    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.getByLabelText("query 0 to key 0: 0.2500")).toHaveTextContent("0.2500");
  });
});
