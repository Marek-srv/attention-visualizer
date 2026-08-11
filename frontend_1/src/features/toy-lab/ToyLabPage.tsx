import { AlertCircle, CheckCircle2, LoaderCircle } from "lucide-react";

import TransformerWorkspace from "../../components/explainer/TransformerWorkspace";
import AdvancedControls from "./AdvancedControls";
import { toInspectRequest, type ToyControlValues } from "./toyControlRequest";
import useToyInspection from "./useToyInspection";
import styles from "./ToyLabPage.module.css";

export type ToyLabPageProps = {
  prompt: string;
  runNonce: number;
  learningMode: "guided" | "explore";
  textbookOpen: boolean;
};

export default function ToyLabPage({ prompt, runNonce, learningMode, textbookOpen }: ToyLabPageProps) {
  const { result, defaults, loading, error, revision, inspect } = useToyInspection(prompt, runNonce);
  const text = prompt.trim() || "I love";

  const recalculate = (values: ToyControlValues): void => {
    void inspect(toInspectRequest(text, values));
  };

  const resetDefaults = (): void => {
    void inspect({ text });
  };

  return (
    <main className={styles.page}>
      <header className={styles.intro}>
        <div>
          <span>Toy Math Lab · exact four-dimensional arithmetic</span>
          <h1>Follow one token through a connected Transformer block</h1>
          <p>Every displayed value comes from the existing FastAPI inspection response. Select a token once, then expand any stage to inspect the real calculation.</p>
        </div>
        <dl>
          <div><dt>Prompt</dt><dd>{text}</dd></div>
          <div><dt>Mode</dt><dd>{learningMode === "guided" ? "Guided forward pass" : "Free exploration"}</dd></div>
        </dl>
      </header>
      {loading && !result ? <div className={styles.status} role="status"><LoaderCircle className={styles.spinner} aria-hidden="true" size={18} /> Inspecting the Toy Transformer…</div> : null}
      {error ? <div className={styles.error} role="alert"><AlertCircle aria-hidden="true" size={18} /><span><strong>Refresh failed.</strong> {error}</span><button type="button" onClick={() => void inspect({ text }, defaults === null)}>Retry</button></div> : null}
      {loading && result ? <div className={styles.refreshing} role="status"><LoaderCircle className={styles.spinner} aria-hidden="true" size={15} /> Recalculating; the last successful visualization remains visible.</div> : null}
      {result ? (
        <>
          <div className={styles.responseBadge}><CheckCircle2 aria-hidden="true" size={15} /> {result.token_count} tokens · vocabulary {result.vocabulary_size} · d_model {result.embedding_dimension}</div>
          <TransformerWorkspace result={result} learningMode={learningMode} textbookOpen={textbookOpen} />
          {defaults ? <AdvancedControls key={revision} result={result} defaults={defaults} busy={loading} onRecalculate={recalculate} onReset={resetDefaults} /> : null}
        </>
      ) : null}
    </main>
  );
}
