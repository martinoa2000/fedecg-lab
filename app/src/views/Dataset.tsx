import { useState } from "react";
import { Bars, ClassName, CountTip, Swatch, useTip } from "../components/charts";
import { CLASS_INFO, formatInt, formatPct, type AppData, type MixRow, type Superclass } from "../data";

const SEQ = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"];

function seqStep(share: number) {
  // share in [0, 1] -> one of six steps, saturating at 60%.
  const step = Math.min(SEQ.length - 1, Math.floor(share * SEQ.length * 1.6));
  return { bg: `var(${SEQ[step]})`, fg: `var(--on-seq-${step})` };
}

function Prevalence({ data }: { data: AppData }) {
  const { distribution } = data.dataset;
  const spread = Math.max(
    ...distribution.map((d) => Math.max(d.train_pct, d.val_pct, d.test_pct) - Math.min(d.train_pct, d.val_pct, d.test_pct)),
  );
  return (
    <div className="panel">
      <p className="chart-title">How common each superclass is</p>
      <p className="chart-sub">
        Share of labelled ECGs. Ticks mark train, validation and test; they differ by at most{" "}
        {spread.toFixed(1)} points, so the splits are comparable.
      </p>
      <Bars
        ariaLabel="Prevalence by superclass"
        max={50}
        data={distribution.map((d) => ({
          key: d.superclass,
          label: <ClassName cls={d.superclass} />,
          value: d.all_pct,
          display: formatPct(d.all_pct),
          color: `var(--${d.superclass})`,
          ticks: [d.train_pct, d.val_pct, d.test_pct],
          tip: (
            <>
              <strong>{CLASS_INFO[d.superclass].name}</strong>
              <br />
              {formatInt(d.all_n)} ECGs, {formatPct(d.all_pct)}
              <br />
              Train {formatPct(d.train_pct)}, validation {formatPct(d.val_pct)}, test {formatPct(d.test_pct)}
            </>
          ),
        }))}
      />
      <p className="legend">
        Records can carry several superclasses, so shares add up to more than 100%.
      </p>
    </div>
  );
}

function Cardinality({ data }: { data: AppData }) {
  const total = data.dataset.records_labeled;
  return (
    <div className="panel">
      <p className="chart-title">Superclasses per ECG</p>
      <p className="chart-sub">
        A quarter of ECGs carry more than one label, which is why the model uses one sigmoid per class
        rather than a softmax.
      </p>
      <Bars
        ariaLabel="Number of superclasses per ECG"
        data={data.dataset.cardinality.map((d) => ({
          key: String(d.n_labels),
          label: `${d.n_labels} ${d.n_labels === 1 ? "label" : "labels"}`,
          value: d.n_records,
          display: formatInt(d.n_records),
          tip: <CountTip title={`${d.n_labels} superclass${d.n_labels === 1 ? "" : "es"}`} n={d.n_records} total={total} />,
        }))}
      />
      <p className="chart-title" style={{ marginTop: 32 }}>
        Most frequent label sets
      </p>
      <Bars
        ariaLabel="Most frequent label combinations"
        data={data.dataset.combinations.map((d) => ({
          key: d.labels,
          label: d.labels.replaceAll("+", " + "),
          value: d.n_records,
          display: formatInt(d.n_records),
          tip: <CountTip title={d.labels.replaceAll("+", " + ")} n={d.n_records} total={total} />,
        }))}
      />
    </div>
  );
}

function Cooccurrence({ data }: { data: AppData }) {
  const { superclasses, cooccurrence } = data.dataset;
  const tip = useTip();
  return (
    <div className="panel">
      <p className="chart-title">Which labels appear together</p>
      <p className="chart-sub">
        Read across a row: of the ECGs with that label, the share that also has the column's label.
        Infarction and conduction disturbance overlap most; normal ECGs almost never overlap.
      </p>
      <div className="heatmap" role="table" aria-label="Label co-occurrence">
        <span role="columnheader" />
        {superclasses.map((c) => (
          <span key={c} className="h-col" role="columnheader">
            {c}
          </span>
        ))}
        {superclasses.map((row, i) => (
          <div key={row} role="row" style={{ display: "contents" }}>
            <span className="h-row" role="rowheader">
              <Swatch cls={row} />
              {row}
            </span>
            {superclasses.map((col, j) => {
              const n = cooccurrence[i][j];
              const share = n / cooccurrence[i][i];
              if (i === j) {
                return (
                  <span key={col} className="cell" data-diag role="cell" {...tip(<CountTip title={`${row} in total`} n={n} total={data.dataset.records_labeled} />)}>
                    {formatInt(n)}
                  </span>
                );
              }
              const { bg, fg } = seqStep(share);
              return (
                <span
                  key={col}
                  className="cell"
                  role="cell"
                  style={{ "--bg": bg, "--fg": fg } as React.CSSProperties}
                  {...tip(
                    <>
                      <strong>
                        {row} with {col}
                      </strong>
                      <br />
                      {formatInt(n)} ECGs, {formatPct(share * 100)} of {row}
                    </>,
                  )}
                >
                  {Math.round(share * 100)}%
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <div className="scale" aria-hidden>
        0%
        <span className="scale-bar">
          {SEQ.map((s) => (
            <i key={s} style={{ background: `var(${s})` }} />
          ))}
        </span>
        60%+
      </div>
    </div>
  );
}

function MixTable<T extends MixRow>({
  rows,
  keyOf,
  labelOf,
  heading,
  limit,
}: {
  rows: T[];
  keyOf: (r: T) => string;
  labelOf: (r: T) => string;
  heading: string;
  limit: number;
}) {
  const [all, setAll] = useState(false);
  const tip = useTip();
  const classes: Superclass[] = ["NORM", "MI", "STTC", "CD", "HYP"];
  const shown = all ? rows : rows.slice(0, limit);
  return (
    <>
      <div className="table-wrap">
        <table className="mix">
          <thead>
            <tr>
              <th scope="col">{heading}</th>
              <th scope="col" className="num">
                ECGs
              </th>
              <th scope="col" style={{ width: "46%" }}>
                <span className="mix-scale">
                  {classes.map((c) => (
                    <span key={c}>
                      <Swatch cls={c} />
                      {c}
                    </span>
                  ))}
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={keyOf(r)} data-small={r.n_records < 100 || undefined}>
                <th scope="row">{labelOf(r)}</th>
                <td className="num">{formatInt(r.n_records)}</td>
                <td>
                  <span className="mix-bars">
                    {classes.map((c) => (
                      <i
                        key={c}
                        role="img"
                        style={{ height: `${Math.max(3, Math.min(100, (r[c] / 80) * 100))}%`, "--c": `var(--${c})` } as React.CSSProperties}
                        {...tip(
                          <>
                            <strong>
                              {labelOf(r)}, {c}
                            </strong>
                            <br />
                            {formatPct(r[c])} of {formatInt(r.n_records)} ECGs
                          </>,
                        )}
                        aria-label={`${c} ${formatPct(r[c])}`}
                      />
                    ))}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > limit && (
        <button type="button" className="button button-outline" onClick={() => setAll((v) => !v)} style={{ justifySelf: "start" }}>
          {all ? `Show top ${limit}` : `Show all ${rows.length}`}
        </button>
      )}
    </>
  );
}

function Demographics({ data }: { data: AppData }) {
  const { age, sex, records_labeled } = data.dataset;
  const top = Math.max(...age.histogram);
  const tip = useTip();
  return (
    <div className="panel">
      <p className="chart-title">Patients</p>
      <p className="chart-sub">Median age {age.median} years. Ages above 89 are withheld by PTB-XL.</p>
      <div className="histogram" aria-label="Age distribution" role="group">
        {age.histogram.map((n, i) => (
          <i
            key={i}
            role="img"
            style={{ height: `${(n / top) * 100}%` }}
            aria-label={`${i * age.bin_width} to ${(i + 1) * age.bin_width - 1} years: ${n}`}
            {...tip(<CountTip title={`${i * age.bin_width}–${(i + 1) * age.bin_width - 1} years`} n={n} total={records_labeled} />)}
          />
        ))}
      </div>
      <div className="histogram-axis tnum">
        <span>0</span>
        <span>30</span>
        <span>60</span>
        <span>90 years</span>
      </div>
      <p className="chart-title" style={{ marginTop: 28 }}>
        Sex
      </p>
      <div className="stack" role="img" aria-label={`${sex.male} male, ${sex.female} female`}>
        <i style={{ flex: sex.male, "--c": "var(--seq-4)" } as React.CSSProperties} />
        <i style={{ flex: sex.female, "--c": "var(--seq-2)" } as React.CSSProperties} />
      </div>
      <p className="legend">
        <span>
          <span className="swatch" style={{ "--c": "var(--seq-4)" } as React.CSSProperties} />
          Male {formatInt(sex.male)}
        </span>
        <span>
          <span className="swatch" style={{ "--c": "var(--seq-2)" } as React.CSSProperties} />
          Female {formatInt(sex.female)}
        </span>
      </p>
    </div>
  );
}

export function Dataset({ data }: { data: AppData }) {
  const { dataset } = data;
  return (
    <div className="page">
      <header className="page-head">
        <h1>PTB-XL</h1>
        <p className="lede">
          {formatInt(dataset.records_labeled)} labelled 12-lead ECGs from {formatInt(dataset.patients)}{" "}
          patients, recorded in Germany between 1989 and 1996. The task is to predict five diagnostic
          superclasses, and one ECG can have several.
        </p>
      </header>

      <section aria-labelledby="labels-title">
        <div className="section-head">
          <h2 id="labels-title">Labels</h2>
          <p>Classes are imbalanced, so results are reported as macro AUROC, which weighs each class equally.</p>
        </div>
        <div className="grid-2">
          <Prevalence data={data} />
          <Cooccurrence data={data} />
        </div>
        <div className="grid-2">
          <Cardinality data={data} />
          <Demographics data={data} />
        </div>
      </section>

      <section aria-labelledby="sources-title">
        <div className="section-head">
          <h2 id="sources-title">Where the ECGs come from</h2>
          <p>
            Recording site and device become the simulated hospitals in phase 5. Their label mix differs a
            lot: one device records mostly normal ECGs, another mostly infarctions.
          </p>
        </div>
        <div style={{ display: "grid", gap: 24 }}>
          <div className="panel" style={{ display: "grid", gap: 16 }}>
            <p className="chart-title">By device</p>
            <MixTable
              rows={dataset.by_device}
              keyOf={(r) => r.device}
              labelOf={(r) => r.device.replace(/\s+/g, " ")}
              heading="Device"
              limit={8}
            />
          </div>
          <div className="panel" style={{ display: "grid", gap: 16 }}>
            <p className="chart-title">By site</p>
            <MixTable
              rows={dataset.by_site}
              keyOf={(r) => String(r.site)}
              labelOf={(r) => `Site ${r.site}`}
              heading="Site"
              limit={8}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
