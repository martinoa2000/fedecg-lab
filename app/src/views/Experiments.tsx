import { useState, type ReactNode } from "react";
import { MixTable } from "../components/MixTable";
import {
  aurocDomain,
  CurveChart,
  DotPlot,
  fmtAuroc,
  fmtDelta,
  type DotRow,
  type Reference,
  type Series,
} from "../components/results";
import { ClassName } from "../components/charts";
import { formatInt, type AppData, type ExperimentRow, type Superclass } from "../data";

const CLASSES: Superclass[] = ["NORM", "MI", "STTC", "CD", "HYP"];

const PARTITIONS: { key: string; label: string; blurb: string }[] = [
  {
    key: "site",
    label: "By site",
    blurb: "One hospital per recording site. Sites 0, 1 and 2 are large; the rest are pooled.",
  },
  {
    key: "device",
    label: "By device",
    blurb: "One hospital per ECG machine. The CS-12 E records 82% normal ECGs; the CS100 3 only 30%.",
  },
  {
    key: "dirichlet",
    label: "Label skew",
    blurb: "Ten hospitals whose label mix is drawn from Dirichlet(0.3): each sees a few classes far more than others.",
  },
];

/** Ordinal blue ramp for "more hospitals": a lighter step is fewer. Starts at
 * step 3, the lightest that keeps 3:1 against the surface. */
const HOSPITAL_STEPS = ["var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];

const PENDING = [
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

function Tip({ row, baseline }: { row: ExperimentRow; baseline?: ExperimentRow }) {
  return (
    <>
      <strong>{fmtAuroc(row.macro_auroc)}</strong> macro AUROC
      <br />
      {row.setting}
      {baseline && row.run !== baseline.run && (
        <>
          <br />
          {fmtDelta(row.macro_auroc - baseline.macro_auroc)} against centralised
        </>
      )}
      {row.communication_mb ? (
        <>
          <br />
          {formatInt(Math.round(row.communication_mb))} MB of weights exchanged
        </>
      ) : null}
    </>
  );
}

function Phase({
  num,
  title,
  count,
  children,
}: {
  num: number;
  title: string;
  count: number;
  children: ReactNode;
}) {
  return (
    <li aria-labelledby={`phase-${num}`}>
      <span className="num" aria-hidden>
        {num}
      </span>
      <div className="body wide">
        <h2 id={`phase-${num}`} style={{ fontSize: "1.375rem" }}>
          {title}
        </h2>
        <span className="status" data-state={count ? "has-results" : "pending"}>
          {count ? `${count} ${count === 1 ? "run" : "runs"}` : "Not run yet"}
        </span>
        {children}
      </div>
    </li>
  );
}

export function Experiments({ data }: { data: AppData }) {
  const { rows, histories, clients, published } = data.experiments;
  const baseline = rows.find((r) => r.run === "centralized");
  const reference = published[0];
  const [partition, setPartition] = useState("site");

  const phase3 = rows.filter((r) => r.phase === 3);
  const phase4 = rows
    .filter((r) => r.phase === 4 && r.partition === "iid")
    .sort((a, b) => (a.n_clients ?? 0) - (b.n_clients ?? 0));
  // Pairs of the same partition sit together: site, device, then label skew.
  const partitionOrder = ["site", "device", "dirichlet"];
  const phase5 = rows
    .filter((r) => r.phase === 5)
    .sort(
      (a, b) =>
        partitionOrder.indexOf(a.partition ?? "") - partitionOrder.indexOf(b.partition ?? "") ||
        (a.algorithm ?? "").localeCompare(b.algorithm ?? ""),
    );
  const phase3Sorted = [...rows.filter((r) => r.phase === 3)].sort((a, b) =>
    a.run === "centralized" ? -1 : b.run === "centralized" ? 1 : 0,
  );
  const ledger = [...phase3Sorted, ...phase4, ...phase5];
  const GROUPS: Record<number, string> = {
    3: "Centralised",
    4: "Federated, IID",
    5: "Federated, non-IID",
  };

  if (!ledger.length) {
    return (
      <div className="page">
        <header className="page-head">
          <h1>Experiments</h1>
          <p className="lede">
            No results yet. Train the centralised baseline, then export again:
          </p>
        </header>
        <div className="empty" style={{ borderStyle: "solid" }}>
          <pre>
            <code>
              uv run python scripts/train_centralized.py{"\n"}uv run python scripts/export_dashboard.py
            </code>
          </pre>
        </div>
      </div>
    );
  }

  const domain = aurocDomain([
    ...ledger.map((r) => r.macro_auroc),
    ...(reference ? [reference.macro_auroc] : []),
  ]);
  const references: Reference[] = [
    ...(baseline ? [{ key: "baseline", value: baseline.macro_auroc, label: "Baseline" }] : []),
    ...(reference ? [{ key: "published", value: reference.macro_auroc, label: "Published" }] : []),
  ];

  const colorOf = (r: ExperimentRow) => {
    if (r.algorithm === "centralized") return r.run === "centralized" ? "var(--ink)" : "var(--muted)";
    if (r.partition === "iid") return HOSPITAL_STEPS[Math.min(phase4.indexOf(r), HOSPITAL_STEPS.length - 1)];
    return "var(--seq-4)";
  };
  const gap = (r: ExperimentRow) =>
    baseline && r.run !== baseline.run ? fmtDelta(r.macro_auroc - baseline.macro_auroc) : "baseline";

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
        hollow: r.algorithm === "fedavg" && r.phase === 5,
        label: r.setting,
        tip: <Tip row={r} baseline={baseline} />,
      },
    ],
  }));

  const curve = (r: ExperimentRow, label = r.setting): Series | null => {
    const h = histories[r.run];
    if (!h?.length) return null;
    return {
      key: r.run,
      label,
      color: colorOf(r),
      best: r.best_step ?? undefined,
      points: h.map((p) => ({ x: p.step, y: p.val_macro_auroc })),
    };
  };

  const pairs = PARTITIONS.map((p) => ({
    ...p,
    avg: phase5.find((r) => r.partition === p.key && r.algorithm === "fedavg"),
    prox: phase5.find((r) => r.partition === p.key && r.algorithm === "fedprox"),
  })).filter((p) => p.avg || p.prox);
  const selected = pairs.find((p) => p.key === partition) ?? pairs[0];
  const selectedClients = selected && clients[(selected.avg ?? selected.prox)!.run];

  return (
    <div className="page">
      <header className="page-head">
        <h1>Experiments</h1>
        <p className="lede">
          Every run is scored on the same test fold with the same model, so a gap between two dots is the cost
          of the one thing that changed.
        </p>
      </header>

      <section aria-labelledby="ledger-title">
        <div className="section-head">
          <h2 id="ledger-title">What each step costs</h2>
          <p>
            Test macro AUROC for every run so far, against the centralised baseline and the best published
            result for the same model family. The axis is zoomed: gridlines are 0.01 apart.
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

      <ol className="phases">
        <Phase num={3} title="Centralised baseline" count={phase3.length}>
          <p>
            A 1D ResNet trained on all eight training folds at once, on random 2.5-second crops with a cosine
            learning-rate schedule. Every other result is measured against it.
          </p>
          {baseline && (
            <>
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
                <div>
                  <p className="chart-title">Test AUROC per superclass</p>
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
                </div>
                <div>
                  <p className="chart-title">Validation macro AUROC per epoch</p>
                  <CurveChart
                    ariaLabel="Validation macro AUROC per epoch for the centralised runs"
                    xLabel="Epoch"
                    series={phase3.map((r) => curve(r)).filter((s): s is Series => s !== null)}
                  />
                </div>
              </div>
            </>
          )}
        </Phase>

        <Phase num={4} title="Federated, IID" count={phase4.length}>
          <p>
            The training set dealt at random to 5, 10 or 20 simulated hospitals, trained with FedAvg. Each
            hospital holds the same label mix, so any gap is the cost of decentralisation alone.
          </p>
          {phase4.length > 0 && (
            <div className="grid-2">
              <div>
                <p className="chart-title">Test macro AUROC by number of hospitals</p>
                <DotPlot
                  ariaLabel="Test macro AUROC by number of hospitals"
                  domain={domain}
                  references={references.filter((r) => r.key === "baseline")}
                  rows={phase4.map((r) => ({
                    key: r.run,
                    label: `${r.n_clients} hospitals`,
                    value: gap(r),
                    points: [
                      {
                        key: r.run,
                        value: r.macro_auroc,
                        color: colorOf(r),
                        label: r.setting,
                        tip: <Tip row={r} baseline={baseline} />,
                      },
                    ],
                  }))}
                />
              </div>
              <div>
                <p className="chart-title">Validation macro AUROC per round</p>
                <CurveChart
                  ariaLabel="Validation macro AUROC per epoch or round, centralised and IID federated runs"
                  xLabel="Round"
                  series={[
                    ...(baseline ? [curve(baseline, "Centralised (per epoch)")] : []),
                    ...phase4.map((r) => curve(r, `${r.n_clients} hospitals`)),
                  ].filter((s): s is Series => s !== null)}
                />
              </div>
            </div>
          )}
        </Phase>

        <Phase num={5} title="Federated, non-IID" count={phase5.length}>
          <p>
            Hospitals built from real metadata (recording site, ECG device) and from synthetic label skew.
            Each partition is trained with FedAvg and with FedProx, which adds a pull toward the global model
            during local training.
          </p>
          {pairs.length > 0 && (
            <>
              <div>
                <p className="chart-title">FedAvg to FedProx, test macro AUROC</p>
                <DotPlot
                  ariaLabel="FedAvg and FedProx test macro AUROC per partition"
                  domain={domain}
                  references={references.filter((r) => r.key === "baseline")}
                  rows={pairs.map((p) => ({
                    key: p.key,
                    label: p.label,
                    value: p.avg && p.prox ? `FedProx ${fmtDelta(p.prox.macro_auroc - p.avg.macro_auroc)}` : "",
                    points: [p.avg, p.prox]
                      .filter((r): r is ExperimentRow => Boolean(r))
                      .map((r) => ({
                        key: r.run,
                        value: r.macro_auroc,
                        color: "var(--seq-4)",
                        hollow: r.algorithm === "fedavg",
                        label: r.setting,
                        tip: <Tip row={r} baseline={baseline} />,
                      })),
                  }))}
                />
                <p className="legend">
                  <span>
                    <i className="dot-key" data-hollow />
                    FedAvg
                  </span>
                  <span>
                    <i className="dot-key" />
                    FedProx
                  </span>
                </p>
              </div>

              <div className="hospitals">
                <div className="hospitals-head">
                  <p className="chart-title">Label mix per hospital</p>
                  <div className="segmented" role="group" aria-label="Partition">
                    {pairs.map((p) => (
                      <button
                        key={p.key}
                        type="button"
                        aria-pressed={p.key === selected.key}
                        onClick={() => setPartition(p.key)}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="chart-sub">{selected.blurb} Bars show each class's share of the hospital's ECGs.</p>
                {selectedClients && (
                  <MixTable
                    rows={selectedClients}
                    keyOf={(r) => r.client}
                    labelOf={(r) => r.client}
                    heading="Hospital"
                    limit={12}
                  />
                )}
              </div>
            </>
          )}
        </Phase>

        {PENDING.map((p) => (
          <Phase key={p.phase} num={p.phase} title={p.title} count={0}>
            <p>{p.body}</p>
            <div className="empty">
              <span>Will show: {p.shows}</span>
            </div>
          </Phase>
        ))}
      </ol>

      <section aria-labelledby="table-title">
        <div className="section-head">
          <h2 id="table-title" style={{ fontSize: "1.25rem" }}>
            All runs
          </h2>
          <p>
            Selected at the epoch or round with the best validation AUROC; F1 uses thresholds tuned on
            validation. Traffic counts weights sent to and from every hospital.
          </p>
        </div>
        <div className="table-wrap">
          <table className="mix runs">
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col" className="num">
                  AUROC
                </th>
                <th scope="col" className="num">
                  Against centralised
                </th>
                <th scope="col" className="num">
                  F1
                </th>
                <th scope="col" className="num">
                  Selected at
                </th>
                <th scope="col" className="num">
                  Traffic
                </th>
              </tr>
            </thead>
            <tbody>
              {ledger.map((r) => (
                <tr key={r.run}>
                  <th scope="row">{r.setting}</th>
                  <td className="num">{fmtAuroc(r.macro_auroc)}</td>
                  <td className="num">{gap(r)}</td>
                  <td className="num">{r.macro_f1 != null ? r.macro_f1.toFixed(3) : "–"}</td>
                  <td className="num">
                    {r.best_step != null
                      ? `${r.algorithm === "centralized" ? "epoch" : "round"} ${r.best_step} of ${r.steps_run}`
                      : "–"}
                  </td>
                  <td className="num">
                    {r.communication_mb ? `${formatInt(Math.round(r.communication_mb))} MB` : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
