import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import Heatmap from "./Heatmap";

describe("Heatmap interaction semantics", () => {
  it("renders exact read-only values without focusable cell controls when selection is unavailable", () => {
    render(
      <Heatmap
        title="Read-only vectors"
        values={[[0.25, -0.5]]}
        rowLabels={["I"]}
        columnLabels={["d0", "d1"]}
      />,
    );

    expect(screen.queryAllByRole("button")).toHaveLength(0);
    const positive = screen.getByLabelText("I, d0: 0.2500");
    const negative = screen.getByLabelText("I, d1: -0.5000");
    expect(positive.tagName).toBe("SPAN");
    expect(negative.tagName).toBe("SPAN");
    expect(positive).toHaveTextContent("0.2500");
    expect(negative).toHaveTextContent("-0.5000");
    expect(positive).toHaveProperty("tabIndex", -1);
  });

  it("renders selectable cells as buttons only when an onSelect handler exists", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<Heatmap title="Selectable" values={[[0.75]]} onSelect={onSelect} />);

    const cell = screen.getByRole("button", { name: "row 0, column 0: 0.7500" });
    await user.click(cell);
    expect(onSelect).toHaveBeenCalledWith({ row: 0, column: 0 });
  });
});
