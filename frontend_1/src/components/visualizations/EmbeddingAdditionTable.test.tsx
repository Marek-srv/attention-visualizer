import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import EmbeddingAdditionTable from "./EmbeddingAdditionTable";
import VectorRow from "./VectorRow";

describe("embedding dimension inspection", () => {
  it("exposes exact numeric addition and selects a dimension on hover or focus", async () => {
    const user = userEvent.setup();
    const select = vi.fn();
    render(
      <EmbeddingAdditionTable
        tokenValues={[0.1, 0.2]}
        positionValues={[1, 2]}
        combinedValues={[1.1, 2.2]}
        selectedDimension={0}
        onSelectDimension={select}
      />,
    );

    expect(screen.getByRole("table", { name: "Exact embedding addition by dimension" })).toBeInTheDocument();
    expect(screen.getByText("= 2.2000")).toBeInTheDocument();
    await user.hover(screen.getByRole("button", { name: "d1" }));
    expect(select).toHaveBeenCalledWith(1);
    await user.tab();
    expect(select).toHaveBeenCalled();
  });

  it("renders a static vector without focusable no-op dimension cells", () => {
    render(<VectorRow label="Static output" values={[0.1, -0.2]} />);
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.getByLabelText("Static output, dimension 1: -0.2000")).toHaveTextContent("-0.2000");
  });
});

