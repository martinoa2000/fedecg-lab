import { Bars } from "../components/charts";
import type { AppData } from "../data";

const PHASES = [
  {
    phase: 3,
    title: "Centralised baseline",
    body: "A 1D ResNet trained on all training folds at once. Every other result is measured against it.",
    shows: "Macro AUROC and per-class F1 on the test fold, next to published PTB-XL baselines.",
  },
  {
    phase: 4,
    title: "Federated, IID",
    body: "The training set split at random across N simulated hospitals, trained with FedAvg.",
    shows: "The AUROC lost to decentralisation alone, as the number of hospitals grows.",
  },
  {
    phase: 5,
    title: "Federated, non-IID",
    body: "Hospitals built from recording site, device and Dirichlet label skew. FedAvg against FedProx.",
    shows: "How much heterogeneity costs, and how much FedProx gets back.",
  },
  {
    phase: 6,
    title: "Differential privacy",
    body: "DP-SGD with Opacus, both centrally and inside each federated client.",
    shows: "AUROC against epsilon: where the privacy guarantee starts to make the model useless.",
  },
  {
    phase: 7,
    title: "Explainability",
    body: "Integrated Gradients and Grad-CAM saliency laid over the raw signal.",
    shows: "Whether infarction predictions rest on the ST segment or on noise.",
  },
];

export function Experiments({ data }: { data: AppData }) {
  const rows = data.experiments;
  return (
    <div className="page">
      <header className="page-head">
        <h1>Experiments</h1>
        <p className="lede">
          Each phase changes one thing against the centralised baseline. Results appear here as each
          phase writes them to <code>results/tables/experiments.csv</code>.
        </p>
      </header>

      <ol className="phases">
        {PHASES.map((p) => {
          const results = rows.filter((r) => r.phase === p.phase);
          return (
            <li key={p.phase} aria-labelledby={`phase-${p.phase}`}>
              <span className="num" aria-hidden>
                {p.phase}
              </span>
              <div className="body">
                <h2 id={`phase-${p.phase}`} style={{ fontSize: "1.375rem" }}>
                  {p.title}
                </h2>
                <span className="status" data-state={results.length ? "has-results" : "pending"}>
                  {results.length ? `${results.length} results` : "Not run yet"}
                </span>
                <p>{p.body}</p>
                {results.length ? (
                  <Bars
                    ariaLabel={`Phase ${p.phase} macro AUROC`}
                    max={1}
                    data={results.map((r, i) => ({
                      key: `${r.setting}-${i}`,
                      label: r.setting,
                      value: r.macro_auroc,
                      display: r.macro_auroc.toFixed(3),
                      color: "var(--seq-4)",
                      tip: (
                        <>
                          <strong>{r.setting}</strong>
                          <br />
                          Macro AUROC {r.macro_auroc.toFixed(3)}
                          {r.epsilon != null && `, epsilon ${r.epsilon}`}
                        </>
                      ),
                    }))}
                  />
                ) : (
                  <div className="empty">
                    <span>Will show: {p.shows}</span>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      <section aria-labelledby="format-title">
        <div className="section-head">
          <h2 id="format-title" style={{ fontSize: "1.25rem" }}>
            Adding results
          </h2>
          <p>
            Append one row per run to <code>results/tables/experiments.csv</code>, then export again. Required
            columns are <code>phase</code>, <code>setting</code> and <code>macro_auroc</code>; add{" "}
            <code>epsilon</code> for phase 6.
          </p>
        </div>
        <div className="empty" style={{ borderStyle: "solid" }}>
          <pre>
            <code>uv run python scripts/export_dashboard.py</code>
          </pre>
        </div>
      </section>
    </div>
  );
}
