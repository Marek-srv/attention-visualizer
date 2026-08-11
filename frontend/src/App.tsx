import { useEffect, useState } from "react";

import { api } from "./api/client";
import Predict from "./components/Predict";
import RealModel from "./components/RealModel";
import ToyMathLab from "./components/ToyMathLab";
import TrainModel from "./components/TrainModel";
import TrainedInspector from "./components/TrainedInspector";

type Mode = "toy" | "train" | "predict" | "inspector" | "real";
type ConnectionState = "checking" | "connected" | "disconnected";

const MODES: Array<{ id: Mode; label: string }> = [
  { id: "toy", label: "Toy Math Lab" },
  { id: "train", label: "Train Model" },
  { id: "predict", label: "Predict" },
  { id: "inspector", label: "Trained Model Inspector" },
  { id: "real", label: "Real Model" },
];

function modeFromHash(): Mode {
  const value = window.location.hash.slice(1);
  return MODES.some((mode) => mode.id === value) ? value as Mode : "toy";
}

export default function App() {
  const [mode, setMode] = useState<Mode>(modeFromHash);
  const [connection, setConnection] = useState<ConnectionState>("checking");

  useEffect(() => {
    const updateMode = () => setMode(modeFromHash());
    window.addEventListener("hashchange", updateMode);
    return () => window.removeEventListener("hashchange", updateMode);
  }, []);

  useEffect(() => {
    let active = true;
    setConnection("checking");
    api.health().then(() => active && setConnection("connected")).catch(() => active && setConnection("disconnected"));
    return () => { active = false; };
  }, []);

  return (
    <main className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <header className="app-header">
        <div className="topbar">
          <a className="brand" href="#toy" onClick={() => setMode("toy")}><span className="brand-mark">A</span><span>Attention Lab</span></a>
          <span className={`status status--${connection}`}><span className="status-dot" aria-hidden="true" />{connection === "checking" ? "Checking backend" : `Backend ${connection}`}</span>
        </div>
        <nav className="mode-nav" aria-label="Learning modes">
          {MODES.map((item) => <a href={`#${item.id}`} className={mode === item.id ? "selected" : ""} aria-current={mode === item.id ? "page" : undefined} onClick={() => setMode(item.id)} key={item.id}>{item.label}</a>)}
        </nav>
      </header>
      <div id="main-content" tabIndex={-1}>
        {mode === "toy" && <ToyMathLab />}
        {mode === "train" && <TrainModel />}
        {mode === "predict" && <Predict />}
        {mode === "inspector" && <TrainedInspector />}
        {mode === "real" && <RealModel />}
      </div>
      <footer className="app-footer"><span>Fixed toy math</span><b aria-hidden="true">·</b><span>Locally trained weights</span><b aria-hidden="true">·</b><span>Optional pretrained weights</span></footer>
    </main>
  );
}
