import { ClassName } from "../../components/charts";
import { aurocDomain, CurveChart, DotPlot, fmtAuroc, type Series } from "../../components/results";
import type { AppData, Superclass } from "../../data";
import { deriveResults, NextPage, NotRunYet, PageHead, RunsTable, runCount } from "./model";

const CLASSES: Superclass[] = ["NORM", "MI", "STTC", "CD", "HYP"];
const TITLE = "Centralised baseline";
const LEDE =
  "A 1D ResNet trained on all eight training folds at once, on random 2.5-second crops with a cosine learning-rate schedule. Every other result is measured against it.";

export function Baseline({ data }: { data: AppData }) {
  const results = deriveResults(data);
  const { baseline, reference, phase3, curve } = results;

  if (!baseline) {
    return <NotRunYet title={TITLE} lede={LEDE} commands={["uv run python scripts/train_centralized.py"]} />;
  }

  return (
    <div className="page">
      <PageHead title={TITLE} status={runCount(phase3.length)} lede={LEDE} />

      <dl className="pair">
        <div>
          <dt>This model</dt>
          <dd>{fmtAuroc(baseline.macro_auroc)}</dd>
        </div>
        {reference && (
          <div>
            <dt>Published {reference.model}</dt>
            <dd>{fmtAuroc(reference.macro_auroc)}</dd>
          </div>
        )}
        {baseline.macro_f1 != null && (
          <div>
            <dt>Macro F1, thresholds tuned on validation</dt>
            <dd>{baseline.macro_f1.toFixed(3)}</dd>
          </div>
        )}
      </dl>

      <div className="grid-2">
        <section aria-labelledby="per-class-title">
          <h2 id="per-class-title" className="chart-title">
            Test AUROC per superclass
          </h2>
          <DotPlot
            ariaLabel="Test AUROC per superclass, centralised baseline"
            domain={aurocDomain(CLASSES.map((c) => baseline[`auroc_${c}`] ?? 0.9))}
            references={[{ key: "macro", value: baseline.macro_auroc, label: "Macro" }]}
            rows={CLASSES.map((c) => {
              const v = baseline[`auroc_${c}`] ?? 0;
              return {
                key: c,
                label: <ClassName cls={c} />,
                value: fmtAuroc(v),
                points: [
                  {
                    key: c,
                    value: v,
                    color: `var(--${c})`,
                    label: c,
                    tip: (
                      <>
                        <strong>{fmtAuroc(v)}</strong> AUROC
                        <br />
                        {c}, centralised baseline
                      </>
                    ),
                  },
                ],
              };
            })}
          />
          <p className="chart-note">
            Hypertrophy is the hardest class: the rarest superclass, and rarely on its own.
          </p>
        </section>
        <section aria-labelledby="curve-title">
          <h2 id="curve-title" className="chart-title">
            Validation macro AUROC per epoch
          </h2>
          <CurveChart
            ariaLabel="Validation macro AUROC per epoch for the centralised runs"
            xLabel="Epoch"
            series={phase3.map((r) => curve(r)).filter((s): s is Series => s !== null)}
          />
        </section>
      </div>

      <section aria-labelledby="runs-title">
        <div className="section-head">
          <h2 id="runs-title">Runs</h2>
          <p>The plain recipe (full-length records, constant learning rate) is rerun, not quoted, so both rows come from the same code and test records.</p>
        </div>
        <RunsTable rows={phase3} results={results} />
      </section>

      <NextPage
        href="#/federated"
        title="Federated"
        question="What does it cost when the training data is split across hospitals?"
      />
    </div>
  );
}
