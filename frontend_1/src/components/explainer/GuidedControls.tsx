import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import type { Dispatch } from "react";

import { stageDefinition } from "../../features/toy-lab/toyAdapter";
import { stageProgress, type ToyExplainerAction, type ToyExplainerState } from "../../features/toy-lab/toyState";
import styles from "./explainer.module.css";

type GuidedControlsProps = {
  state: ToyExplainerState;
  dispatch: Dispatch<ToyExplainerAction>;
};

export default function GuidedControls({ state, dispatch }: GuidedControlsProps) {
  const progress = stageProgress(state.stage);
  return (
    <div className={styles.guided} aria-label="Guided forward-pass controls">
      <div className={styles.playbackButtons}>
        <button type="button" aria-label="Previous stage" onClick={() => dispatch({ type: "previous-stage" })} disabled={progress === 0}><SkipBack aria-hidden="true" size={16} /></button>
        <button type="button" aria-label={state.isPlaying ? "Pause guided playback" : "Play guided playback"} onClick={() => dispatch({ type: state.isPlaying ? "pause" : "play" })}>
          {state.isPlaying ? <Pause aria-hidden="true" size={16} /> : <Play aria-hidden="true" size={16} />}
        </button>
        <button type="button" aria-label="Next stage" onClick={() => dispatch({ type: "next-stage" })} disabled={progress === 1}><SkipForward aria-hidden="true" size={16} /></button>
        <button type="button" aria-label="Restart guided playback" onClick={() => dispatch({ type: "restart" })}><RotateCcw aria-hidden="true" size={15} /></button>
      </div>
      <div className={styles.progressText}>
        <strong>{stageDefinition(state.stage).title}</strong>
        <progress value={progress} max={1}>{Math.round(progress * 100)}%</progress>
      </div>
      <label className={styles.speed}>
        <span>Speed</span>
        <select value={state.speed} onChange={(event) => dispatch({ type: "set-speed", speed: Number(event.currentTarget.value) as 0.5 | 1 | 1.5 })}>
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="1.5">1.5×</option>
        </select>
      </label>
    </div>
  );
}

