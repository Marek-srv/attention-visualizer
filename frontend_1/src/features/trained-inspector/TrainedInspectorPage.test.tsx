import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CausalMaskTable } from "./TrainedInspectorPage";

describe("trained attention causal-mask view", () => {
  it("labels backend false values as selectable masked future positions with zero weight", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <CausalMaskTable
        allowed={[[true, false], [true, true]]}
        rowLabels={["I", "love"]}
        columnLabels={["I", "love"]}
        selected={{ row: 0, column: 0 }}
        onSelect={onSelect}
      />,
    );

    const visible = screen.getByRole("button", { name: "I to I: Visible" });
    const future = screen.getByRole("button", { name: "I to love: Masked future key, probability zero" });
    expect(visible).toHaveAttribute("data-masked", "false");
    expect(future).toHaveAttribute("data-masked", "true");
    expect(future).toHaveTextContent("Masked");
    expect(future).toHaveTextContent("weight 0");

    await user.click(future);
    expect(onSelect).toHaveBeenCalledWith({ row: 0, column: 1 });
  });
});
