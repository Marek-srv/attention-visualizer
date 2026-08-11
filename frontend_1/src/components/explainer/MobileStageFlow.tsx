import type { Dispatch } from "react";
import { ChevronDown, LockKeyhole } from "lucide-react";

import type { ToyInspectResponse } from "../../types/api";
import { stageVector, TOY_STAGES } from "../../features/toy-lab/toyAdapter";
import type { ToyExplainerAction, ToyExplainerState } from "../../features/toy-lab/toyState";
import styles from "./explainer.module.css";

type MobileStageFlowProps = {
  result: ToyInspectResponse;
  state: ToyExplainerState;
  dispatch: Dispatch<ToyExplainerAction>;
};

export default function MobileStageFlow({ result, state, dispatch }: MobileStageFlowProps) {
  const token = result.tokens[state.selectedTokenPosition];
  return (
    <div className={styles.mobileFlow} aria-label="Vertical Transformer stage flow">
      <p><span className={styles.flowSwatch} /> Constant-width connector means data flow, not magnitude.</p>
      {TOY_STAGES.map((stage, index) => {
        const vector = stageVector(result, stage.id, state.selectedTokenPosition);
        const available = stage.available(result);
        return (
          <div className={styles.mobileStageWrap} key={stage.id}>
            {index > 0 ? <div className={styles.mobileConnector}><ChevronDown aria-hidden="true" size={18} /></div> : null}
            <button
              type="button"
              className={styles.mobileStage}
              aria-current={state.stage === stage.id ? "step" : undefined}
              onClick={() => dispatch({ type: "select-stage", stage: stage.id })}
            >
              <span className={styles.mobileIndex}>{String(index + 1).padStart(2, "0")}</span>
              <span><strong>{stage.shortLabel}</strong><small>{stage.title}</small></span>
              <span className={styles.mobileValues}>
                {vector?.slice(0, 4).map((value, dimension) => <em key={dimension}>{value.toFixed(2)}</em>)}
                {!available ? <em><LockKeyhole aria-hidden="true" size={12} /> Trained mode</em> : null}
              </span>
              <span className={styles.mobileToken}>{token?.token ?? "—"}</span>
            </button>
          </div>
        );
      })}
    </div>
  );
}

