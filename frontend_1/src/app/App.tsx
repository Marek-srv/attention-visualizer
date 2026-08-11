import { lazy, Suspense, useEffect, useMemo, useState } from "react";

import { api } from "../api/endpoints";
import AppShell, { type ConnectionState } from "./AppShell";
import { routeFromHash, type AppMode } from "./routes";
import { WorkspaceContext, type LearningMode } from "./workspaceContext";

const ToyLabPage = lazy(() => import("../features/toy-lab/ToyLabPage"));
const TrainingPage = lazy(() => import("../features/training/TrainingPage"));
const PredictionPage = lazy(() => import("../features/prediction/PredictionPage"));
const TrainedInspectorPage = lazy(() => import("../features/trained-inspector/TrainedInspectorPage"));
const RealModelPage = lazy(() => import("../features/real-model/RealModelPage"));

function ModeLoading() {
  return <section className="mode-loading" role="status"><span className="loading-pulse" aria-hidden="true" /><strong>Preparing this learning mode…</strong></section>;
}

export default function App() {
  const [mode, setMode] = useState<AppMode>(() => routeFromHash(window.location.hash));
  const [prompt, setPrompt] = useState("I love");
  const [runNonce, setRunNonce] = useState(1);
  const [learningMode, setLearningMode] = useState<LearningMode>("guided");
  const [textbookOpen, setTextbookOpen] = useState(true);
  const [connection, setConnection] = useState<ConnectionState>("checking");

  useEffect(() => {
    const onHashChange = () => setMode(routeFromHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api.health({ signal: controller.signal })
      .then(() => setConnection("connected"))
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setConnection("disconnected");
      });
    return () => controller.abort();
  }, [runNonce]);

  useEffect(() => {
    const closeDetail = (event: KeyboardEvent) => {
      if (event.key === "Escape" && textbookOpen) setTextbookOpen(false);
    };
    window.addEventListener("keydown", closeDetail);
    return () => window.removeEventListener("keydown", closeDetail);
  }, [textbookOpen]);

  const contextValue = useMemo(() => ({
    mode,
    prompt,
    runNonce,
    learningMode,
    textbookOpen,
    setPrompt,
    requestRun: () => setRunNonce((value) => value + 1),
  }), [learningMode, mode, prompt, runNonce, textbookOpen]);

  return (
    <WorkspaceContext.Provider value={contextValue}>
      <AppShell
        mode={mode}
        prompt={prompt}
        learningMode={learningMode}
        textbookOpen={textbookOpen}
        connection={connection}
        onModeChange={setMode}
        onPromptChange={setPrompt}
        onRun={() => setRunNonce((value) => value + 1)}
        onLearningModeChange={setLearningMode}
        onTextbookToggle={() => setTextbookOpen((value) => !value)}
      >
        <Suspense fallback={<ModeLoading />}>
          {mode === "toy" && <ToyLabPage prompt={prompt} runNonce={runNonce} learningMode={learningMode} textbookOpen={textbookOpen} />}
          {mode === "training" && <TrainingPage />}
          {mode === "prediction" && <PredictionPage />}
          {mode === "trained-inspector" && <TrainedInspectorPage />}
          {mode === "real-model" && <RealModelPage />}
        </Suspense>
      </AppShell>
    </WorkspaceContext.Provider>
  );
}
