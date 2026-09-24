import type { AppData } from "../../data";
import { PrivacyResults } from "../PrivacyAndTrust";
import { deriveResults, NextPage, NotRunYet, PageHead, RunsTable, runCount } from "./model";

const TITLE = "Differential privacy";
const LEDE =
  "DP-SGD with Opacus: every record's gradient is clipped and noise is added to each step, so the model provably depends little on any single ECG. Trained centrally and inside each of five federated hospitals, at three privacy budgets.";

export function Privacy({ data }: { data: AppData }) {
  const results = deriveResults(data);
  const { baseline, phase6 } = results;

  if (!phase6.length) {
    return (
      <NotRunYet
        title={TITLE}
        lede={LEDE}
        commands={["uv run python scripts/train_centralized.py --config dp_eps8.yaml"]}
      />
    );
  }

  return (
    <div className="page">
      <PageHead title={TITLE} status={runCount(phase6.length)} lede={LEDE} />
      <section aria-label="Cost of privacy">
        <PrivacyResults rows={phase6} baseline={baseline} />
      </section>
      <section aria-labelledby="dp-runs-title">
        <div className="section-head">
          <h2 id="dp-runs-title">Runs</h2>
          <p>
            Each privacy setting has a twin with clipping and noise switched off: the DP recipe (large batches,
            30 epochs or rounds) costs AUROC by itself, so the price of privacy is the gap to that twin.
          </p>
        </div>
        <RunsTable rows={phase6} results={results} />
      </section>
      <NextPage
        href="#/explainability"
        title="Explainability"
        question="When the model flags an infarction, what part of the ECG is it looking at?"
      />
    </div>
  );
}
