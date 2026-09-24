import { useState } from "react";
import { ClassName, Swatch } from "../components/charts";
import { SaliencyStrip } from "../components/Saliency";
import { DotPlot, EpsilonChart, fmtAuroc, fmtDelta, type EpsilonSeries } from "../components/results";
import type { ExperimentRow, Explain, Segment, Superclass } from "../data";

const CLASSES: Superclass[] = ["NORM", "MI", "STTC", "CD", "HYP"];

/* ---------------------------------------------------------------- phase 6 */

function epsilonTip(r: ExperimentRow, control?: ExperimentRow) {
  return (
    <>
      <strong>{fmtAuroc(r.macro_auroc)}</strong> macro AUROC
      <br />
      {r.setting}
      {r.epsilon != null && (
        <>
          <br />
          ε spent {r.epsilon.toFixed(2)} at δ = 10⁻⁵
        </>
      )}
      {control && r.run !== control.run && (
        <>
          <br />
          {fmtDelta(r.macro_auroc - control.macro_auroc)} against the same recipe without privacy
        </>
      )}
    </>
  );
}

export function PrivacyResults({ rows, baseline }: { rows: ExperimentRow[]; baseline?: ExperimentRow }) {
  const central = rows.filter((r) => r.algorithm === "centralized");
  const federated = rows.filter((r) => r.algorithm !== "centralized");
  const controlOf = (group: ExperimentRow[]) => group.find((r) => r.epsilon == null);

  const series: EpsilonSeries[] = [
    { key: "central", label: "Centralised", color: "var(--ink)", group: central },
    { key: "federated", label: "FedAvg, 5 hospitals", color: "var(--seq-4)", group: federated },
  ]
    .filter((s) => s.group.length)
    .map(({ group, ...s }) => ({
      ...s,
      points: group.map((r) => ({
        epsilon: r.epsilon == null ? null : Math.round(r.epsilon),
        value: r.macro_auroc,
        tip: epsilonTip(r, controlOf(group)),
      })),
    }));

  if (!series.length) return null;
  const strictest = central
    .filter((r) => r.epsilon != null)
    .sort((a, b) => (a.epsilon ?? 0) - (b.epsilon ?? 0))[0];
  const centralControl = controlOf(central);

  return (
    <>
      {strictest && centralControl && (
        <dl className="pair">
          <div>
            <dt>Same recipe, no privacy</dt>
            <dd>{fmtAuroc(centralControl.macro_auroc)}</dd>
          </div>
          <div>
            <dt>DP-SGD, ε = {Math.round(strictest.epsilon!)}</dt>
            <dd>{fmtAuroc(strictest.macro_auroc)}</dd>
          </div>
          <div>
            <dt>Cost of the guarantee</dt>
            <dd>{fmtDelta(strictest.macro_auroc - centralControl.macro_auroc)}</dd>
          </div>
        </dl>
      )}
      <div>
        <p className="chart-title">Test macro AUROC by privacy budget</p>
        <EpsilonChart
          ariaLabel="Test macro AUROC against privacy budget epsilon, centralised and federated"
          series={series}
          references={baseline ? [{ key: "baseline", value: baseline.macro_auroc, label: "Baseline" }] : []}
        />
        <p className="chart-note">
          ε bounds how much any single ECG can change what the model does, with probability 1 − δ, δ = 10⁻⁵.
          In the federated runs each hospital protects its own records at that ε. The baseline line is the
          phase 3 model, trained with its own, non-private recipe.
        </p>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- phase 7 */

const SEGMENT_LABEL: Record<Segment, string> = {
  qrs: "QRS complex",
  st: "ST segment",
  t: "T wave",
  other: "Rest of the beat",
};

export function ExplainResults({ explain }: { explain: Explain }) {
  const available = CLASSES.filter((c) => explain.examples.some((e) => e.superclass === c));
  const [cls, setCls] = useState<Superclass>(available.includes("MI") ? "MI" : (available[0] ?? "MI"));
  if (!explain.segments.length) return null;

  const example = explain.examples.find((e) => e.superclass === cls);
  const leadNames = explain.lead_names ?? [];
  const leadII = leadNames.indexOf("II");
  const topLead = example
    ? example.attribution
        .map((lead, i) => [lead.reduce((a, b) => a + b, 0), i] as const)
        .sort((a, b) => b[0] - a[0])[0][1]
    : -1;
  const shownLeads = [leadII, topLead].filter((l, i, all) => l >= 0 && all.indexOf(l) === i);

  const enrichment = (method: string, segment: Segment) =>
    explain.segments.find((r) => r.superclass === cls && r.method === method && r.segment === segment)
      ?.enrichment ?? null;
  const segments: Segment[] = ["qrs", "st", "t", "other"];
  const sanity = explain.sanity.find((r) => r.superclass === cls);
  // Where the random-weight Grad-CAM is zero for most records, its enrichment
  // rests on a handful of maps and would mislead; it is left out.
  const showRandom = (sanity?.grad_cam_random_nonzero ?? 0) >= 0.5;
  const shown = explain.segments.filter(
    (r) => r.method === "grad_cam" || (showRandom && r.method === "grad_cam_random"),
  );
  const maxEnrichment = Math.max(2, ...shown.map((r) => r.enrichment ?? 0));

  const leadRow = explain.leads.find((r) => r.superclass === cls);
  const normRow = explain.leads.find((r) => r.superclass === "NORM");
  const leadValues = explain.leads.flatMap((r) => leadNames.map((l) => r[l] ?? 0));

  return (
    <>
      <div className="hospitals-head">
        <p className="chart-title">Explained class</p>
        <div className="segmented" role="group" aria-label="Superclass">
          {available.map((c) => (
            <button key={c} type="button" aria-pressed={c === cls} onClick={() => setCls(c)}>
              <Swatch cls={c} />
              {c}
            </button>
          ))}
        </div>
      </div>

      {example && (
        <div>
          <p className="chart-title">
            One test ECG the model flags as <ClassName cls={cls} long />, probability {example.probability.toFixed(2)}
          </p>
          {shownLeads.map((lead) => (
            <SaliencyStrip
              key={lead}
              example={example}
              lead={lead}
              leadName={leadNames[lead] ?? String(lead)}
              fs={explain.fs ?? 100}
            />
          ))}
          <p className="legend strip-legend">
            <span>
              <i className="st-key" />
              ST segment, delineated in lead II
            </span>
            <span>
              <i className="attr-key" />
              Integrated Gradients, darker is more
            </span>
          </p>
        </div>
      )}

      <div className="grid-2">
        <div>
          <p className="chart-title">Where Grad-CAM looks, against chance</p>
          <DotPlot
            ariaLabel={`Grad-CAM enrichment per beat segment for ${cls}, trained and random-weight model`}
            domain={[0, Math.ceil(maxEnrichment * 2) / 2]}
            tickStep={0.5}
            format={(v) => `${v.toFixed(2)} times chance`}
            metric="enrichment"
            references={[{ key: "chance", value: 1, label: "Chance" }]}
            rows={segments.map((segment) => {
              const trained = enrichment("grad_cam", segment);
              const random = showRandom ? enrichment("grad_cam_random", segment) : null;
              return {
                key: segment,
                label: SEGMENT_LABEL[segment],
                value: trained == null ? "–" : `${trained.toFixed(2)}×`,
                points: [
                  ...(random == null
                    ? []
                    : [
                        {
                          key: "random",
                          value: random,
                          color: "var(--muted)",
                          hollow: true,
                          label: `${SEGMENT_LABEL[segment]}, random weights`,
                          tip: (
                            <>
                              <strong>{random.toFixed(2)}×</strong> chance
                              <br />
                              Same architecture, random weights
                            </>
                          ),
                        },
                      ]),
                  ...(trained == null
                    ? []
                    : [
                        {
                          key: "trained",
                          value: trained,
                          color: `var(--${cls})`,
                          label: `${SEGMENT_LABEL[segment]}, trained model`,
                          tip: (
                            <>
                              <strong>{trained.toFixed(2)}×</strong> chance
                              <br />
                              {SEGMENT_LABEL[segment]}, trained model, {cls}
                            </>
                          ),
                        },
                      ]),
                ],
              };
            })}
          />
          <p className="legend">
            <span>
              <i className="dot-key" style={{ "--c": `var(--${cls})` } as React.CSSProperties} />
              Trained model
            </span>
            <span>
              <i className="dot-key" data-hollow style={{ "--c": "var(--muted)" } as React.CSSProperties} />
              Random weights
            </span>
          </p>
          <p className="chart-note">
            Share of Grad-CAM attention in each segment divided by the share of time the segment covers, over{" "}
            {explain.segments.find((r) => r.superclass === cls)?.n_records ?? 0} correctly detected test ECGs.
            1 is chance.
            {!showRandom &&
              ` The random-weight Grad-CAM for this class is zero on ${Math.round(
                (1 - (sanity?.grad_cam_random_nonzero ?? 0)) * 100,
              )}% of records, so it is not shown.`}
          </p>
        </div>

        {leadRow && (
          <div>
            <p className="chart-title">Which leads carry the attribution</p>
            <DotPlot
              ariaLabel={`Share of attribution per lead for ${cls}, against NORM`}
              domain={[
                Math.floor(Math.min(...leadValues) * 100) / 100,
                Math.ceil(Math.max(...leadValues) * 100) / 100,
              ]}
              tickStep={0.01}
              format={(v) => `${(v * 100).toFixed(1)}%`}
              metric="share of attribution"
              references={[{ key: "even", value: 1 / 12, label: "Even" }]}
              rows={leadNames.map((lead) => {
                const v = leadRow[lead] ?? 0;
                const norm = normRow?.[lead];
                return {
                  key: lead,
                  label: lead,
                  value: `${(v * 100).toFixed(1)}%`,
                  points: [
                    ...(norm != null && cls !== "NORM"
                      ? [
                          {
                            key: "norm",
                            value: norm,
                            color: "var(--muted)",
                            hollow: true,
                            label: `Lead ${lead}, NORM`,
                            tip: (
                              <>
                                <strong>{(norm * 100).toFixed(1)}%</strong> of attribution
                                <br />
                                Lead {lead}, normal ECGs
                              </>
                            ),
                          },
                        ]
                      : []),
                    {
                      key: "cls",
                      value: v,
                      color: `var(--${cls})`,
                      label: `Lead ${lead}, ${cls}`,
                      tip: (
                        <>
                          <strong>{(v * 100).toFixed(1)}%</strong> of attribution
                          <br />
                          Lead {lead}, {cls}
                        </>
                      ),
                    },
                  ],
                };
              })}
            />
            {cls !== "NORM" && (
              <p className="legend">
                <span>
                  <i className="dot-key" style={{ "--c": `var(--${cls})` } as React.CSSProperties} />
                  {cls}
                </span>
                <span>
                  <i className="dot-key" data-hollow style={{ "--c": "var(--muted)" } as React.CSSProperties} />
                  NORM
                </span>
              </p>
            )}
          </div>
        )}
      </div>

      {sanity && (
        <dl className="pair">
          <div>
            <dt>Integrated Gradients, trained vs. random weights</dt>
            <dd>
              {sanity.integrated_gradients != null ? `ρ = ${sanity.integrated_gradients.toFixed(2)}` : "Not defined"}
            </dd>
          </div>
          <div>
            <dt>Grad-CAM, trained vs. random weights</dt>
            <dd>{sanity.grad_cam != null ? `ρ = ${sanity.grad_cam.toFixed(2)}` : "Not defined"}</dd>
          </div>
        </dl>
      )}
      <p className="chart-note">
        Rank correlation between maps from the trained model and from the same network with random weights
        (Adebayo et al., 2018). Near 0 means the map depends on what the model learned; near 1 means it mostly
        shows the shape of the input. Not defined when the random-weight maps are all zero.
      </p>
    </>
  );
}
