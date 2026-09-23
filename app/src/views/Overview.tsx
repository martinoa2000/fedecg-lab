import { useState } from "react";
import { EcgPaper } from "../components/EcgPaper";
import { Swatch } from "../components/charts";
import { CLASS_INFO, formatInt, type AppData, type Superclass } from "../data";

const QUESTIONS = [
  {
    title: "What does decentralisation cost?",
    body: "AUROC lost when one training set becomes N hospitals running FedAvg.",
    phase: "Phases 3 and 4",
  },
  {
    title: "What does heterogeneity cost?",
    body: "Hospitals differ in devices and patient mix. How far does that pull performance down, and does FedProx recover it?",
    phase: "Phase 5",
  },
  {
    title: "What does privacy cost?",
    body: "DP-SGD gives a formal guarantee. At what epsilon does the model stop being useful?",
    phase: "Phase 6",
  },
  {
    title: "Can the model be trusted?",
    body: "When it flags an infarction, is it looking at the ST segment or at noise?",
    phase: "Phase 7",
  },
];

export function Overview({ data }: { data: AppData }) {
  const { dataset, signals } = data;
  const [cls, setCls] = useState<Superclass>("MI");
  const record = signals.records.find((r) => r.ecg_id === signals.examples[cls][0])!;

  return (
    <div className="page">
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-text">
          <h1 id="hero-title">What does it cost to learn from ECGs that never leave the hospital?</h1>
          <p className="lede">
            fedecg-lab trains a 12-lead ECG classifier centrally, then across simulated hospitals,
            then under differential privacy, and measures the accuracy given up at each step.
          </p>
        </div>

        <div className="controls">
          <div className="segmented" role="group" aria-label="Diagnostic superclass">
            {dataset.superclasses.map((c) => (
              <button key={c} type="button" aria-pressed={c === cls} onClick={() => setCls(c)}>
                <Swatch cls={c} />
                {c}
              </button>
            ))}
          </div>
          <p className="class-note">
            <strong>{CLASS_INFO[cls].name}.</strong> {CLASS_INFO[cls].blurb}
          </p>
        </div>

        <figure className="ecg-frame" style={{ margin: 0 }}>
          <EcgPaper
            signal={record.filtered}
            leads={signals.leads}
            fs={signals.fs}
            animate
            label={`12-lead ECG ${record.ecg_id}, labelled ${record.labels.join(" and ")}`}
          />
          <figcaption className="ecg-caption">
            <span>
              ECG {record.ecg_id}, {record.sex}
              {record.age !== null && `, ${record.age} years`}, band-passed {dataset.bandpass_hz[0]}–
              {dataset.bandpass_hz[1]} Hz
            </span>
            <span>25 mm/s, 10 mm/mV</span>
          </figcaption>
        </figure>
      </section>

      <dl className="facts">
        <div>
          <dt>Labelled ECGs</dt>
          <dd>
            {formatInt(dataset.records_labeled)}
            <small>
              of {formatInt(dataset.records_total)};{" "}
              {formatInt(dataset.records_total - dataset.records_labeled)} have no diagnostic statement
            </small>
          </dd>
        </div>
        <div>
          <dt>Patients</dt>
          <dd>
            {formatInt(dataset.patients)}
            <small>each kept within one fold</small>
          </dd>
        </div>
        <div>
          <dt>Per recording</dt>
          <dd>
            12 leads, 10 s
            <small>{dataset.sampling_rate_hz} Hz, 1,000 samples per lead</small>
          </dd>
        </div>
        <div>
          <dt>Train, validation, test</dt>
          <dd className="tnum">
            {formatInt(dataset.split.train)} / {formatInt(dataset.split.val)} /{" "}
            {formatInt(dataset.split.test)}
            <small>official folds 1–8, 9 and 10</small>
          </dd>
        </div>
      </dl>

      <section aria-labelledby="questions-title">
        <div className="section-head">
          <h2 id="questions-title">Four costs, measured side by side</h2>
          <p>Each experiment changes one thing against the centralised baseline.</p>
        </div>
        <ol className="questions">
          {QUESTIONS.map((q) => (
            <li key={q.title}>
              <h3>{q.title}</h3>
              <p>{q.body}</p>
              <span className="phase">{q.phase}</span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
