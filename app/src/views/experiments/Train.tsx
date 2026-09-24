import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type ExperimentConfig, type Run, type TrainOptions } from "../../api";
import { aurocDomain, CurveChart, DotPlot, fmtAuroc, fmtDelta, type Series } from "../../components/results";
import { formatInt, type AppData, type TuningRow } from "../../data";
import { PageHead } from "./model";

/** The phase 3 recipe (configs/centralized.yaml): the form starts from it. */
const DEFAULTS: TrainOptions = {
  base_channels: 64,
  crop_samples: 250,
  epochs: 50,
  learning_rate: 0.001,
  loss: "bce",
  seeds: 1,
  amplitude: 0.1,
  noise: 0.05,
  wander: 0.1,
  lead_dropout: 0.1,
};

const AUGMENT: { key: "amplitude" | "noise" | "wander" | "lead_dropout"; label: string; hint: string }[] = [
  { key: "amplitude", label: "Gain", hint: "Scale each lead by 1 ± this" },
  { key: "noise", label: "Noise", hint: "Gaussian noise, standardized units" },
  { key: "wander", label: "Baseline wander", hint: "Slow drift up to this amplitude" },
  { key: "lead_dropout", label: "Lead dropout", hint: "Chance each lead is zeroed" },
];

const STATUS_LABEL: Record<Run["status"], string> = {
  queued: "Queued",
  running: "Training",
  stopping: "Stopping",
  done: "Finished",
  failed: "Failed",
  stopped: "Stopped",
};

const active = (r: Run) => r.status === "queued" || r.status === "running" || r.status === "stopping";

/* ------------------------------------------------------------- the study */

function TuningStudy({ rows }: { rows: TuningRow[] }) {
  const reference = rows.find((r) => r.run === "tune_baseline");
  // The study's rows are named after its configs; anything else was started
  // from this page and is listed with them.
  const sorted = [...rows].sort((a, b) => b.val_macro_auroc - a.val_macro_auroc);
  const domain = aurocDomain(rows.map((r) => r.val_macro_auroc), 0.005);
  return (
    <>
      <DotPlot
        ariaLabel="Validation macro AUROC per training setting"
        domain={domain}
        tickStep={0.005}
        references={reference ? [{ key: "ref", value: reference.val_macro_auroc, label: "Before tuning" }] : []}
        rows={sorted.map((r) => ({
          key: r.run,
          label: r.setting,
          value: reference && r.run !== reference.run ? fmtDelta(r.val_macro_auroc - reference.val_macro_auroc) : "reference",
          points: [
            {
              key: r.run,
              value: r.val_macro_auroc,
              color: r.run === reference?.run ? "var(--ink)" : "var(--seq-4)",
              label: r.setting,
              tip: (
                <>
                  <strong>{fmtAuroc(r.val_macro_auroc)}</strong> validation macro AUROC
                  <br />
                  {r.setting}
                  <br />
                  {formatInt(r.n_parameters)} parameters, {r.seeds > 1 ? `${r.seeds} seeds, ` : ""}
                  {Math.round(r.seconds / 60)} min
                </>
              ),
            },
          ],
        }))}
      />
      <div className="table-wrap" style={{ marginTop: 20 }}>
        <table className="mix runs">
          <thead>
            <tr>
              <th scope="col">Setting</th>
              <th scope="col" className="num">
                Val AUROC
              </th>
              <th scope="col" className="num">
                HYP
              </th>
              <th scope="col">Augmentation</th>
              <th scope="col">Loss</th>
              <th scope="col" className="num">
                Parameters
              </th>
              <th scope="col" className="num">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.run}>
                <th scope="row">{r.setting}</th>
                <td className="num">{fmtAuroc(r.val_macro_auroc)}</td>
                <td className="num">{r.val_auroc_HYP != null ? fmtAuroc(r.val_auroc_HYP) : "–"}</td>
                <td>{r.augment.replaceAll("_", " ")}</td>
                <td>{r.loss.replace("_", " ")}</td>
                <td className="num">{formatInt(r.n_parameters)}</td>
                <td className="num">{Math.max(1, Math.round(r.seconds / 60))} min</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

/* ------------------------------------------------------------ the forms */

function Choice<T extends string | number | null>({
  label,
  value,
  values,
  format,
  onChange,
}: {
  label: string;
  value: T;
  values: T[];
  format?: (v: T) => string;
  onChange: (v: T) => void;
}) {
  return (
    <div className="field">
      <span className="field-label" id={`f-${label}`}>
        {label}
      </span>
      <div className="segmented" role="group" aria-labelledby={`f-${label}`}>
        {values.map((v) => (
          <button key={String(v)} type="button" aria-pressed={v === value} onClick={() => onChange(v)}>
            {format ? format(v) : String(v)}
          </button>
        ))}
      </div>
    </div>
  );
}

function TuneForm({ onStart, busy }: { onStart: (o: TrainOptions) => Promise<void>; busy: boolean }) {
  const [options, setOptions] = useState<TrainOptions>(DEFAULTS);
  const set = <K extends keyof TrainOptions>(key: K, value: TrainOptions[K]) =>
    setOptions((o) => ({ ...o, [key]: value }));
  // Rough duration on an Apple M-series laptop: ~2.5 s per epoch at 32
  // channels, about linear in width, times the number of seeds.
  const minutes = Math.max(1, Math.round((options.epochs * 2.5 * (options.base_channels / 32) * options.seeds) / 60));

  return (
    <form
      className="panel train-form"
      onSubmit={(e) => {
        e.preventDefault();
        void onStart(options);
      }}
    >
      <h3>Try a setting</h3>
      <p className="chart-sub">
        Scored on the validation fold only, never on test, so you can try as many as you like. Starts from the
        phase 3 recipe.
      </p>
      <Choice label="Network width" value={options.base_channels} values={[16, 32, 64, 96]} format={(v) => `${v} ch`} onChange={(v) => set("base_channels", v)} />
      <Choice
        label="Training crops"
        value={options.crop_samples}
        values={[250, 500, null]}
        format={(v) => (v === null ? "Whole record" : `${v / 100} s`)}
        onChange={(v) => set("crop_samples", v)}
      />
      <Choice
        label="Loss"
        value={options.loss}
        values={["bce", "weighted_bce", "focal"] as TrainOptions["loss"][]}
        format={(v) => ({ bce: "Cross-entropy", weighted_bce: "Class-weighted", focal: "Focal" })[v]}
        onChange={(v) => set("loss", v)}
      />
      <Choice label="Ensemble" value={options.seeds} values={[1, 2, 3, 5]} format={(v) => (v === 1 ? "1 model" : `${v} seeds`)} onChange={(v) => set("seeds", v)} />
      <Choice
        label="Learning rate"
        value={options.learning_rate}
        values={[0.0003, 0.001, 0.003]}
        format={(v) => String(v)}
        onChange={(v) => set("learning_rate", v)}
      />
      <label className="field">
        <span className="field-label">Maximum epochs</span>
        <input
          type="number"
          inputMode="numeric"
          min={1}
          max={100}
          value={options.epochs}
          onChange={(e) => set("epochs", Math.min(100, Math.max(1, Number(e.target.value) || 1)))}
        />
      </label>
      <fieldset className="field augment">
        <legend className="field-label">Augmentation</legend>
        {AUGMENT.map((a) => (
          <label key={a.key} className="slider">
            <span>
              {a.label} <small>{a.hint}</small>
            </span>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.05}
              value={options[a.key]}
              onChange={(e) => set(a.key, Number(e.target.value))}
            />
            <output>{options[a.key].toFixed(2)}</output>
          </label>
        ))}
      </fieldset>
      <div className="form-foot">
        <button type="submit" className="button button-primary">
          {busy ? "Add to queue" : "Start training"}
        </button>
        <span className="chart-sub">About {minutes} min</span>
      </div>
    </form>
  );
}

function ExperimentForm({
  experiments,
  onStart,
}: {
  experiments: ExperimentConfig[];
  onStart: (name: string) => Promise<void>;
}) {
  const [name, setName] = useState("centralized");
  const [confirming, setConfirming] = useState(false);
  const chosen = experiments.find((e) => e.name === name);
  return (
    <div className="panel train-form">
      <h3>Rerun an experiment</h3>
      <p className="chart-sub">
        Trains an existing config exactly as the scripts do and scores the test fold. Its row in the results is
        replaced.
      </p>
      <label className="field">
        <span className="field-label">Config</span>
        <select value={name} onChange={(e) => (setName(e.target.value), setConfirming(false))}>
          {[3, 4, 5, 6].map((phase) => (
            <optgroup key={phase} label={`Phase ${phase}`}>
              {experiments
                .filter((e) => e.phase === phase)
                .map((e) => (
                  <option key={e.name} value={e.name}>
                    {e.setting}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </label>
      {chosen && <p className="chart-note">configs/{chosen.name}.yaml, run by scripts/{chosen.script}.py</p>}
      <div className="form-foot">
        {confirming ? (
          <>
            <button
              type="button"
              className="button button-primary"
              onClick={() => {
                setConfirming(false);
                void onStart(name);
              }}
            >
              Replace the result
            </button>
            <button type="button" className="button button-outline" onClick={() => setConfirming(false)}>
              Keep it
            </button>
          </>
        ) : (
          <button type="button" className="button button-outline" onClick={() => setConfirming(true)}>
            Rerun
          </button>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- the runs */

function RunDetail({ run }: { run: Run }) {
  const series: Series[] = useMemo(() => {
    const members = [...new Set(run.progress.map((p) => p.member))];
    const colors = ["var(--seq-4)", "var(--seq-3)", "var(--seq-5)", "var(--ink)", "var(--muted)"];
    return members.map((m, i) => ({
      key: String(m),
      label: members.length > 1 ? `Seed ${m}` : "Validation",
      color: colors[i % colors.length],
      points: run.progress.filter((p) => p.member === m).map((p) => ({ x: p.step, y: p.val_macro_auroc })),
    }));
  }, [run.progress]);
  return (
    <div className="run-detail">
      {series.length > 0 && series[0].points.length > 1 ? (
        <CurveChart
          series={series}
          xLabel={run.config.startsWith("fed") ? "Round" : "Epoch"}
          ariaLabel={`Validation macro AUROC per epoch for ${run.label}`}
          height={200}
        />
      ) : (
        <p className="chart-sub">The curve appears after the second epoch.</p>
      )}
      {run.log && (
        <pre className="run-log" aria-label="Training output">
          {run.log.slice(-14).join("\n")}
        </pre>
      )}
    </div>
  );
}

function RunList({
  runs,
  selected,
  onSelect,
  onStop,
}: {
  runs: Run[];
  selected: string | null;
  onSelect: (id: string) => void;
  onStop: (id: string) => void;
}) {
  return (
    <ul className="run-list">
      {runs.map((r) => {
        const last = r.progress[r.progress.length - 1];
        const best = r.progress.length ? Math.max(...r.progress.map((p) => p.val_macro_auroc)) : null;
        return (
          <li key={r.id} data-selected={r.id === selected || undefined}>
            <button type="button" className="run-row" onClick={() => onSelect(r.id)} aria-expanded={r.id === selected}>
              <span className="run-status" data-status={r.status}>
                {STATUS_LABEL[r.status]}
              </span>
              <span className="run-label">
                {r.label}
                <small>{r.kind === "tune" ? "validation only" : "test fold"}</small>
              </span>
              <span className="run-progress">
                {r.result != null
                  ? `${r.kind === "tune" ? "val" : "test"} ${fmtAuroc(r.result)}`
                  : last
                    ? `${r.members > 1 ? `seed ${r.member}/${r.members}, ` : ""}epoch ${last.step}, best ${fmtAuroc(best!)}`
                    : ""}
              </span>
            </button>
            {active(r) && r.status !== "stopping" && (
              <button type="button" className="button button-outline run-stop" onClick={() => onStop(r.id)}>
                {r.status === "queued" ? "Remove" : "Stop"}
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/* --------------------------------------------------------------- the page */

export function Train({ data }: { data: AppData }) {
  const tuning = data.experiments.tuning.rows;
  const [online, setOnline] = useState<boolean | null>(null);
  const [experiments, setExperiments] = useState<ExperimentConfig[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [finishedSinceLoad, setFinishedSinceLoad] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const list = await api.runs();
      setRuns((previous) => {
        const justFinished = previous.some(
          (p) => active(p) && list.find((r) => r.id === p.id && r.status === "done"),
        );
        if (justFinished) setFinishedSinceLoad(true);
        return list;
      });
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  // Connect, then poll: quickly while something trains, slowly otherwise.
  useEffect(() => {
    void refresh();
    api.experiments().then(setExperiments, () => {});
  }, [refresh]);
  const training = runs.some(active);
  useEffect(() => {
    const id = window.setInterval(() => void refresh(), training ? 2000 : 5000);
    return () => window.clearInterval(id);
  }, [refresh, training]);
  useEffect(() => {
    if (online && !experiments.length) api.experiments().then(setExperiments, () => {});
  }, [online, experiments.length]);

  const selectedRun = runs.find((r) => r.id === selected);
  useEffect(() => {
    if (!selected) return setDetail(null);
    let cancelled = false;
    const load = () => api.run(selected).then((r) => !cancelled && setDetail(r), () => {});
    void load();
    const id = selectedRun && active(selectedRun) ? window.setInterval(load, 2000) : undefined;
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [selected, selectedRun?.status, selectedRun]);

  const start = async (submit: () => Promise<Run>) => {
    setError(null);
    try {
      const run = await submit();
      setSelected(run.id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="page">
      <PageHead
        title="Tuning and training"
        lede="Which training choices raise the baseline, measured on the validation fold only. With the local training server running, try your own settings or rerun an experiment from here."
      />

      <section aria-labelledby="study-title">
        <div className="section-head">
          <h2 id="study-title">What helped on validation</h2>
          <p>
            Each setting changes one thing in the phase 3 recipe (the last combines two) and is scored on fold 9.
            The test fold is untouched, so these can be compared freely; a winner still has to be trained for
            real and tested once.
          </p>
        </div>
        {tuning.length ? (
          <div className="panel">
            <TuningStudy rows={tuning} />
          </div>
        ) : (
          <div className="empty">
            <span>No tuning results yet. Run the study from the repository root, then export again:</span>
            <pre>
              <code>{"uv run python scripts/tune.py --config tune_wide.yaml\nuv run python scripts/export_dashboard.py"}</code>
            </pre>
          </div>
        )}
      </section>

      <section aria-labelledby="train-title">
        <div className="section-head">
          <h2 id="train-title">Train on this machine</h2>
          <p className="server-status" data-online={online ?? undefined}>
            {online === null
              ? "Looking for the training server…"
              : online
                ? training
                  ? "Training server connected. One run trains at a time; new ones wait in the queue."
                  : "Training server connected and idle."
                : "The training server is not running."}
          </p>
        </div>

        {online === false && (
          <div className="empty">
            <span>Start it from the repository root and keep that terminal open; this page connects on its own:</span>
            <pre>
              <code>uv run python scripts/serve.py</code>
            </pre>
          </div>
        )}

        {online && (
          <>
            <div className="grid-2 train-grid">
              <TuneForm busy={training} onStart={(o) => start(() => api.tune(o))} />
              <ExperimentForm experiments={experiments} onStart={(n) => start(() => api.experiment(n))} />
            </div>
            {error && (
              <p className="form-error" role="alert">
                Could not start the run: {error}
              </p>
            )}
            {finishedSinceLoad && (
              <p className="notice" role="status">
                A run finished and the dashboard data was exported again.{" "}
                <button type="button" className="button button-outline" onClick={() => window.location.reload()}>
                  Reload to see it
                </button>
              </p>
            )}
            {runs.length > 0 && (
              <div className="runs-panel">
                <h3>Runs this session</h3>
                <RunList
                  runs={runs}
                  selected={selected}
                  onSelect={(id) => setSelected((s) => (s === id ? null : id))}
                  onStop={(id) => void api.stop(id).then(refresh, () => {})}
                />
                {detail && selected === detail.id && <RunDetail run={detail} />}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
