import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import ProbabilityBars from "./ProbabilityBars";

describe("ProbabilityBars", () => {
  it("orders model probabilities from highest to lowest without mutating the response", () => {
    const predictions = [
      { token: "low", token_id: 1, logit: -1, probability: 0.1 },
      { token: "high", token_id: 2, logit: 2, probability: 0.7 },
      { token: "middle", token_id: 3, logit: 0.5, probability: 0.2 },
    ] as const;

    render(<ProbabilityBars predictions={predictions} />);

    const rows = screen.getAllByRole("listitem");
    expect(rows).toHaveLength(3);
    expect(within(rows[0]!).getByText("high")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("middle")).toBeInTheDocument();
    expect(within(rows[2]!).getByText("low")).toBeInTheDocument();
    expect(predictions.map((prediction) => prediction.token)).toEqual(["low", "high", "middle"]);
  });
});
