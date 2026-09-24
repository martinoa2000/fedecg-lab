import { useState } from "react";
import { MixTable } from "../../components/MixTable";
import { CurveChart, DotPlot, fmtDelta, type Series } from "../../components/results";
import type { AppData, ExperimentRow } from "../../data";
import { deriveResults, NextPage, NotRunYet, PageHead, RunsTable, RunTip, runCount } from "./model";

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

const TITLE = "Federated learning";
const LEDE =
  "The training set split across simulated hospitals that never share an ECG, only model weights. First at random, to isolate the cost of splitting; then by real recording site and device, and by synthetic label skew.";

export function Federated({ data }: { data: AppData }) {
  const results = deriveResults(data);
  const { baseline, phase4, phase5, references, colorOf, gap, curve, domainFor } = results;
  // Both charts share one axis, zoomed to this page's runs, so gaps compare.
  const domain = domainFor([...phase4, ...phase5]);
  const [partition, setPartition] = useState("site");
  const baselineRef = references.filter((r) => r.key === "baseline");

  if (!phase4.length && !phase5.length) {
    return (
      <NotRunYet
        title={TITLE}
        lede={LEDE}
        commands={[
          "uv run python scripts/train_federated.py --config fedavg_iid_10.yaml",
          "uv run python scripts/train_federated.py --config fedavg_site.yaml",
        ]}
      />
    );
  }

  const pairs = PARTITIONS.map((p) => ({
    ...p,
    avg: phase5.find((r) => r.partition === p.key && r.algorithm === "fedavg"),
    prox: phase5.find((r) => r.partition === p.key && r.algorithm === "fedprox"),
  })).filter((p) => p.avg || p.prox);
  const selected = pairs.find((p) => p.key === partition) ?? pairs[0];
  const selectedClients = selected && data.experiments.clients[(selected.avg ?? selected.prox)!.run];

  return (
    <div className="page">
      <PageHead title={TITLE} status={runCount(phase4.length + phase5.length)} lede={LEDE} />

      {phase4.length > 0 && (
        <section aria-labelledby="iid-title">
          <div className="section-head">
            <h2 id="iid-title">Random hospitals</h2>
            <p>
              The training set dealt at random to 5, 10 or 20 hospitals, trained with FedAvg. Each hospital
              holds the same label mix, so any gap is the cost of decentralisation alone. A round costs one pass
              over the data, like an epoch.
            </p>
          </div>
          <div className="grid-2">
            <div>
              <p className="chart-title">Test macro AUROC by number of hospitals</p>
              <DotPlot
                ariaLabel="Test macro AUROC by number of hospitals"
                domain={domain}
                references={baselineRef}
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
                      tip: <RunTip row={r} baseline={baseline} />,
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
        </section>
      )}

      {pairs.length > 0 && (
        <section aria-labelledby="noniid-title">
          <div className="section-head">
            <h2 id="noniid-title">Hospitals that differ</h2>
            <p>
              Hospitals built from real metadata and from synthetic label skew. Each partition is trained with
              FedAvg and with FedProx, which adds a pull toward the global model during local training.
            </p>
          </div>
          <div>
            <p className="chart-title">FedAvg to FedProx, test macro AUROC</p>
            <DotPlot
              ariaLabel="FedAvg and FedProx test macro AUROC per partition"
              domain={domain}
              references={baselineRef}
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
                    tip: <RunTip row={r} baseline={baseline} />,
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
        </section>
      )}

      <section aria-labelledby="fed-runs-title">
        <div className="section-head">
          <h2 id="fed-runs-title">Runs</h2>
          <p>Traffic counts the weights sent to and from every hospital over all rounds run.</p>
        </div>
        <RunsTable rows={[...phase4, ...phase5]} results={results} />
      </section>

      <NextPage
        href="#/privacy"
        title="Privacy"
        question="Model updates still leak. What does a formal privacy guarantee cost?"
      />
    </div>
  );
}
