import { createContext, useContext } from "react";

import type { AppMode } from "./routes";

export type LearningMode = "guided" | "explore";

export type WorkspaceContextValue = {
  mode: AppMode;
  prompt: string;
  runNonce: number;
  learningMode: LearningMode;
  textbookOpen: boolean;
  setPrompt: (value: string) => void;
  requestRun: () => void;
};

export const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

export function useWorkspace(): WorkspaceContextValue {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error("useWorkspace must be used inside WorkspaceContext.Provider");
  return value;
}

