import { useState } from "react";
import { EcgPaper, type EcgLayout } from "../components/EcgPaper";
import { ClassName, Swatch } from "../components/charts";
import { CLASS_INFO, type AppData } from "../data";

export function Signals({ data }: { data: AppData }) {
  const { signals, dataset } = data;
  const [ecgId, setEcgId] = useState(signals.examples.MI[0]);
  const [filtered, setFiltered] = useState(true);
  const [layout, setLayout] = useState<EcgLayout>("standard");
  const record = signals.records.find((r) => r.ecg_id === ecgId)!;
  const codes = Object.entries(record.scp_codes).sort((a, b) => b[1] - a[1]);

  return (
    <div className="page">
      <header className="page-head">
        <h1>Signals</h1>
        <p className="lede">
          Example ECGs with a single superclass each, human-validated where possible. Compare the raw
          recording with the {dataset.bandpass_hz[0]}–{dataset.bandpass_hz[1]} Hz band-pass the model is
          trained on: it removes baseline wander and muscle noise.
        </p>
      </header>

      <div className="viewer">
        <nav className="record-list" aria-label="Example ECGs">
          {dataset.superclasses.map((cls) => (
            <div key={cls} className="record-group">
              <h3>
                <Swatch cls={cls} />
                {CLASS_INFO[cls].name}
              </h3>
              <ul>
                {signals.examples[cls].map((id) => {
                  const r = signals.records.find((x) => x.ecg_id === id)!;
                  return (
                    <li key={id}>
                      <button
                        type="button"
                        className="button"
                        aria-pressed={id === ecgId}
                        onClick={() => setEcgId(id)}
                      >
                        <span>ECG {id}</span>
                        <span className="meta">
                          {r.sex === "male" ? "M" : "F"}
                          {r.age !== null && `, ${r.age}`}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>

        <div className="record-detail">
          <div className="controls">
            <div className="segmented" role="group" aria-label="Signal">
              <button type="button" aria-pressed={filtered} onClick={() => setFiltered(true)}>
                Band-passed
              </button>
              <button type="button" aria-pressed={!filtered} onClick={() => setFiltered(false)}>
                Raw
              </button>
            </div>
            <div className="segmented" role="group" aria-label="Layout">
              <button type="button" aria-pressed={layout === "standard"} onClick={() => setLayout("standard")}>
                Standard printout
              </button>
              <button type="button" aria-pressed={layout === "stacked"} onClick={() => setLayout("stacked")}>
                All leads, 10 s
              </button>
            </div>
          </div>

          <figure className="ecg-frame" style={{ margin: 0 }}>
            <EcgPaper
              signal={filtered ? record.filtered : record.raw}
              leads={signals.leads}
              fs={signals.fs}
              layout={layout}
              label={`ECG ${record.ecg_id}, ${filtered ? "band-passed" : "raw"}, labelled ${record.labels.join(" and ")}`}
            />
            <figcaption className="ecg-caption">
              <span>
                ECG {record.ecg_id}, {filtered ? "band-passed" : "raw signal"}
              </span>
              <span>25 mm/s, 10 mm/mV, one small square is 40 ms by 0.1 mV</span>
            </figcaption>
          </figure>

          <dl className="record-facts">
            <div>
              <dt>Label</dt>
              <dd>
                {record.labels.map((l) => (
                  <ClassName key={l} cls={l} long />
                ))}
              </dd>
            </div>
            <div>
              <dt>Patient</dt>
              <dd>
                {record.sex === "male" ? "Male" : "Female"}
                {record.age !== null ? `, ${record.age} years` : ", age withheld"}
              </dd>
            </div>
            <div>
              <dt>Device</dt>
              <dd>{record.device.replace(/\s+/g, " ")}</dd>
            </div>
            <div>
              <dt>Fold</dt>
              <dd>
                {record.strat_fold} (
                {record.strat_fold === dataset.split_folds.test
                  ? "test"
                  : record.strat_fold === dataset.split_folds.val
                    ? "validation"
                    : "train"}
                )
              </dd>
            </div>
            <div>
              <dt>Reviewed by a cardiologist</dt>
              <dd>{record.validated ? "Yes" : "No"}</dd>
            </div>
          </dl>

          <div className="report">
            <p className="control-label">Original report, in German</p>
            <blockquote lang="de">{record.report}</blockquote>
          </div>

          <div className="report">
            <p className="control-label">SCP-ECG statements and likelihood</p>
            <ul className="codes">
              {codes.map(([code, likelihood]) => (
                <li key={code}>
                  {code}
                  <span>{likelihood > 0 ? `${likelihood}%` : "no likelihood given"}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
