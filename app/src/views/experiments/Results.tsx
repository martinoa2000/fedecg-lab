import { DotPlot, fmtAuroc, fmtDelta, type DotRow } from "../../components/results";
import type { AppData, ExperimentRow } from "../../data";
import { deriveResults, GROUPS, NotRunYet, PageHead, RunsTable, RunTip, runCount } from "./model";

const range = (rows: ExperimentRow[], baseline: number) => {
  const gaps = rows.map((r) => r.macro_auroc - baseline);
  return gaps.length ? `${fmtDelta(Math.max(...gaps))} to ${fmtDelta(Math.min(...gaps))}` : null;
};

export function Results({ data }: { data: AppData }) {
  const results = deriveResults(data);
  const { baseline, reference, ledger, domain, references, colorOf, gap, phase4, phase5, phase6 } = results;

  if (!ledger.length) {
    return (
      <NotRunYet
        title="Results"
        lede="Every experiment, scored on the same test fold, against the centralised baseline."
        commands={["uv run python scripts/train_centralized.py"]}
      />
    );
  }

  const ledgerRows: DotRow[] = ledger.map((r) => ({
    key: r.run,
    group: GROUPS[r.phase],
    label: r.setting,
    value: gap(r),
    points: [
      {
        key: r.run,
        value: r.macro_auroc,
        color: colorOf(r),
        // Hollow marks the reference member of a pair: FedAvg against
        // FedProx, and the no-privacy control against its DP runs.
        hollow: (r.algorithm === "fedavg" && r.phase === 5) || (r.phase === 6 && r.epsilon == null),
        label: r.setting,
        tip: <RunTip row={r} baseline={baseline} />,
      },
    ],
  }));

  const b = baseline?.macro_auroc ?? 0;
  const privateRuns = phase6.filter((r) => r.epsilon != null);
  const steps = [
    {
      href: "#/baseline",
      title: "Baseline",
      figure: baseline ? fmtAuroc(baseline.macro_auroc) : "–",
      caption: "centralised test macro AUROC",
    },
    {
      href: "#/federated",
      title: "Federated",
      figure: baseline ? range([...phase4, ...phase5], b) : null,
      caption: "across 4 to 20 simulated hospitals",
    },
    {
      href: "#/privacy",
      title: "Privacy",
      figure: baseline ? range(privateRuns, b) : null,
      caption: "with DP-SGD at ε = 8 down to 1",
    },
    {
      href: "#/explainability",
      title: "Explainability",
      figure: data.explain.examples.length ? `${data.explain.examples.length} classes` : null,
      caption: "traced to the QRS, ST and T segments",
    },
  ];

  return (
    <div className="page">
      <PageHead
        title="Results"
        status={runCount(ledger.length)}
        lede="Every run is scored on the same test fold with the same model, so a gap between two dots is the cost of the one thing that changed."
      />

      <ol className="steps">
        {steps.map((s) => (
          <li key={s.href}>
            <a href={s.href}>
              <span className="step-title">{s.title}</span>
              <strong>{s.figure ?? "Not run yet"}</strong>
              <span className="step-caption">{s.figure ? s.caption : "Run the experiments to see this step"}</span>
            </a>
          </li>
        ))}
      </ol>

      <section aria-labelledby="ledger-title">
        <div className="section-head">
          <h2 id="ledger-title">What each step costs</h2>
          <p>
            Test macro AUROC for every run, against the centralised baseline and the best published result for
            the same model family. The axis is zoomed: gridlines are 0.01 apart.
          </p>
        </div>
        <div className="panel">
          <DotPlot rows={ledgerRows} references={references} domain={domain} ariaLabel="Test macro AUROC per run" />
          {reference && (
            <p className="chart-note">
              Published: {reference.model}, {fmtAuroc(reference.macro_auroc)} ({reference.source}).
            </p>
          )}
        </div>
      </section>

      <section aria-labelledby="table-title">
        <div className="section-head">
          <h2 id="table-title">All runs</h2>
          <p>
            Selected at the epoch or round with the best validation AUROC; F1 uses thresholds tuned on
            validation. Traffic counts weights sent to and from every hospital; ε is the privacy budget spent
            (δ = 10⁻⁵), largest over hospitals for federated runs.
          </p>
        </div>
        <RunsTable rows={ledger} results={results} />
      </section>
    </div>
  );
}
