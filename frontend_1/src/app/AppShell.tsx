import type { ReactNode } from "react";
import { BookOpen, CircleHelp, Play, Sparkles, Wifi, WifiOff } from "lucide-react";

import { APP_ROUTES, hashForMode, type AppMode } from "./routes";
import type { LearningMode } from "./workspaceContext";

const attributionUrl = new URL("../../THIRD_PARTY_NOTICES.md", import.meta.url).href;

export type ConnectionState = "checking" | "connected" | "disconnected";

type AppShellProps = {
  mode: AppMode;
  prompt: string;
  learningMode: LearningMode;
  textbookOpen: boolean;
  connection: ConnectionState;
  onModeChange: (mode: AppMode) => void;
  onPromptChange: (value: string) => void;
  onRun: () => void;
  onLearningModeChange: (mode: LearningMode) => void;
  onTextbookToggle: () => void;
  children: ReactNode;
};

export default function AppShell({
  mode,
  prompt,
  learningMode,
  textbookOpen,
  connection,
  onModeChange,
  onPromptChange,
  onRun,
  onLearningModeChange,
  onTextbookToggle,
  children,
}: AppShellProps) {
  const activeDescription = APP_ROUTES.find((route) => route.id === mode)?.description
    ?? "Interactive Transformer workspace";

  return (
    <div className="interactive-app">
      <a className="skip-link" href="#workspace-main">Skip to model workspace</a>
      <header className="control-bar">
        <div className="product-lockup">
          <span className="product-mark" aria-hidden="true"><Sparkles size={19} /></span>
          <span><strong>Attention Lab</strong><small>Interactive</small></span>
        </div>

        <label className="top-field mode-field">
          <span>Mode</span>
          <select
            value={mode}
            onChange={(event) => {
              const next = event.currentTarget.value as AppMode;
              onModeChange(next);
              window.location.hash = hashForMode(next);
            }}
          >
            {APP_ROUTES.map((route) => <option value={route.id} key={route.id}>{route.label}</option>)}
          </select>
        </label>

        <label className="top-field prompt-field">
          <span>Prompt</span>
          <input
            value={prompt}
            maxLength={500}
            onChange={(event) => onPromptChange(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.nativeEvent.isComposing) onRun();
            }}
            aria-describedby="active-mode-description"
          />
        </label>

        <button className="run-button" type="button" disabled={!prompt.trim()} onClick={onRun}>
          <Play size={16} aria-hidden="true" />
          {mode === "training" ? "Refresh" : mode === "prediction" ? "Use prompt" : "Run / inspect"}
        </button>

        <div className="learning-toggle" aria-label="Learning navigation mode">
          <button type="button" className={learningMode === "guided" ? "selected" : ""} aria-pressed={learningMode === "guided"} onClick={() => onLearningModeChange("guided")}>Guided</button>
          <button type="button" className={learningMode === "explore" ? "selected" : ""} aria-pressed={learningMode === "explore"} onClick={() => onLearningModeChange("explore")}>Explore</button>
        </div>

        <button className={`icon-button ${textbookOpen ? "selected" : ""}`} type="button" aria-pressed={textbookOpen} aria-label="Toggle contextual textbook" onClick={onTextbookToggle}>
          {textbookOpen ? <BookOpen size={18} /> : <CircleHelp size={18} />}
        </button>

        <div className={`connection-state ${connection}`} role="status" aria-live="polite">
          {connection === "connected" ? <Wifi size={15} /> : <WifiOff size={15} />}
          <span>{connection === "checking" ? "Checking backend" : connection === "connected" ? "Backend connected" : "Backend disconnected"}</span>
        </div>
      </header>

      <nav className="mode-ribbon" aria-label="Application modes">
        {APP_ROUTES.map((route) => (
          <a
            href={hashForMode(route.id)}
            className={mode === route.id ? "selected" : ""}
            aria-current={mode === route.id ? "page" : undefined}
            onClick={() => onModeChange(route.id)}
            key={route.id}
          >
            <strong>{route.shortLabel}</strong>
            <span>{route.description}</span>
          </a>
        ))}
      </nav>

      <p className="visually-hidden" id="active-mode-description">{activeDescription}</p>
      <div id="workspace-main" tabIndex={-1}>{children}</div>
      <footer className="interactive-footer">
        <span>Flow-line width shows the data route unless an attention view explicitly labels probability encoding.</span>
        <a href={attributionUrl}>Attribution</a>
      </footer>
    </div>
  );
}
