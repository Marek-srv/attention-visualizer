import { BookOpen, TriangleAlert } from "lucide-react";

import type { ToyStageId } from "../../features/toy-lab/toyState";
import Formula from "../visualizations/Formula";
import { STAGE_LESSONS } from "./stageContent";
import styles from "./TextbookPanel.module.css";

type TextbookPanelProps = {
  stage: ToyStageId;
  formula: string;
  mobile?: boolean;
};

export default function TextbookPanel({ stage, formula, mobile = false }: TextbookPanelProps) {
  const lesson = STAGE_LESSONS[stage];
  return (
    <aside className={styles.panel} data-mobile={mobile ? "true" : "false"} aria-labelledby="stage-textbook-title">
      <div className={styles.heading}>
        <BookOpen aria-hidden="true" size={18} />
        <div>
          <span>Contextual textbook</span>
          <h2 id="stage-textbook-title">{lesson.heading}</h2>
        </div>
      </div>
      <section>
        <h3>What this stage does</h3>
        <p>{lesson.does}</p>
      </section>
      <section>
        <h3>Current formula</h3>
        <Formula expression={formula} label={`${lesson.heading} formula`} />
      </section>
      <section>
        <h3>What these values mean</h3>
        <p>{lesson.values}</p>
      </section>
      <section className={styles.notice}>
        <h3>What to notice</h3>
        <p>{lesson.notice}</p>
      </section>
      {lesson.warning ? (
        <section className={styles.warning}>
          <h3><TriangleAlert aria-hidden="true" size={16} /> Interpretation note</h3>
          <p>{lesson.warning}</p>
        </section>
      ) : null}
    </aside>
  );
}

