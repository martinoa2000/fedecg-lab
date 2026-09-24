import type { ReactNode } from "react";
import { aurocDomain, fmtAuroc, fmtDelta, type Reference, type Series } from "../../components/results";
import { formatInt, type AppData, type ExperimentRow } from "../../data";

/**
 * Everything the experiment pages derive from `experiments.json`, computed in
 * one place so every page sorts, colors and measures runs the same way.
 */
export interface Results {
  rows: ExperimentRow[];
  baseline?: ExperimentRow;
  reference?: AppData["experiments"]["published"][number];
  phase3: ExperimentRow[];
  phase4: ExperimentRow[];
  phase5: ExperimentRow[];
  phase6: ExperimentRow[];
  /** Every run, in reading order: phase, then the order within each phase. */
  ledger: ExperimentRow[];
  /** One zoomed AUROC axis over every run and the published result. */
  domain: [number, number];
  references: Reference[];
  colorOf: (r: ExperimentRow) => string;
  /** Signed gap to the centralised baseline, or "baseline" for itself. */
  gap: (r: ExperimentRow) => string;
  curve: (r: ExperimentRow, label?: string) => Series | null;
  /** A zoomed axis for a subset of runs, always including the baseline. */
  domainFor: (rows: ExperimentRow[]) => [number, number];
}

/** Ordinal blue ramp for "more hospitals": a lighter step is fewer. Starts at
 * step 3, the lightest that keeps 3:1 against the surface. */
const HOSPITAL_STEPS = ["var(--seq-3)", "var(--seq-4)", "var(--seq-5)"];

const PARTITION_ORDER = ["site", "device", "dirichlet"];

export const GROUPS: Record<number, string> = {
  3: "Centralised",
  4: "Federated, IID",
  5: "Federated, non-IID",
  6: "Differential privacy",
};

export function deriveResults(data: AppData): Results {
  const { rows, histories, published } = data.experiments;
  const baseline = rows.find((r) => r.run === "centralized");
  const reference = published[0];

  const phase3 = rows
    .filter((r) => r.phase === 3)
    // The baseline first, then the other recipes from best to worst.
    .sort((a, b) =>
      a.run === "centralized" ? -1 : b.run === "centralized" ? 1 : b.macro_auroc - a.macro_auroc,
    );
  const phase4 = rows
    .filter((r) => r.phase === 4 && r.partition === "iid")
    .sort((a, b) => (a.n_clients ?? 0) - (b.n_clients ?? 0));
  // Pairs of the same partition sit together: site, device, then label skew.
  const phase5 = rows
    .filter((r) => r.phase === 5)
    .sort(
      (a, b) =>
        PARTITION_ORDER.indexOf(a.partition ?? "") - PARTITION_ORDER.indexOf(b.partition ?? "") ||
        (a.algorithm ?? "").localeCompare(b.algorithm ?? ""),
    );
  // Centralised before federated; strictest budget first, no-privacy control last.
  const phase6 = rows
    .filter((r) => r.phase === 6)
    .sort(
      (a, b) =>
        (a.algorithm === "centralized" ? 0 : 1) - (b.algorithm === "centralized" ? 0 : 1) ||
        (a.epsilon ?? Infinity) - (b.epsilon ?? Infinity),
    );
  const ledger = [...phase3, ...phase4, ...phase5, ...phase6];

  const domain = ledger.length
    ? aurocDomain([...ledger.map((r) => r.macro_auroc), ...(reference ? [reference.macro_auroc] : [])])
    : ([0.8, 0.95] as [number, number]);
  const references: Reference[] = [
    ...(baseline ? [{ key: "baseline", value: baseline.macro_auroc, label: "Baseline" }] : []),
    ...(reference ? [{ key: "published", value: reference.macro_auroc, label: "Published" }] : []),
  ];

  const colorOf = (r: ExperimentRow) => {
    if (r.phase === 6) return r.algorithm === "centralized" ? "var(--ink)" : "var(--seq-4)";
    if (r.algorithm === "centralized") return r.run === "centralized" ? "var(--ink)" : "var(--muted)";
    if (r.partition === "iid") return HOSPITAL_STEPS[Math.min(phase4.indexOf(r), HOSPITAL_STEPS.length - 1)];
    return "var(--seq-4)";
  };
  const gap = (r: ExperimentRow) =>
    baseline && r.run !== baseline.run ? fmtDelta(r.macro_auroc - baseline.macro_auroc) : "baseline";
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

  const domainFor = (subset: ExperimentRow[]) =>
    aurocDomain([...subset, ...(baseline ? [baseline] : [])].map((r) => r.macro_auroc));

  return {
    domainFor,
    rows,
    baseline,
    reference,
    phase3,
    phase4,
    phase5,
    phase6,
    ledger,
    domain,
    references,
    colorOf,
    gap,
    curve,
  };
}

export function RunTip({ row, baseline }: { row: ExperimentRow; baseline?: ExperimentRow }) {
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

/** Heading block shared by the experiment pages. */
export function PageHead({ title, lede, status }: { title: string; lede: ReactNode; status?: string }) {
  return (
    <header className="page-head">
      <h1>{title}</h1>
      {status && (
        <span className="status" data-state="has-results">
          {status}
        </span>
      )}
      <p className="lede">{lede}</p>
    </header>
  );
}

/** What a page shows before its experiments have been run and exported. */
export function NotRunYet({ title, lede, commands }: { title: string; lede: ReactNode; commands: string[] }) {
  return (
    <div className="page">
      <PageHead title={title} lede={lede} />
      <div className="empty">
        <span>No results yet. Run this from the repository root, then export again:</span>
        <pre>
          <code>{[...commands, "uv run python scripts/export_dashboard.py"].join("\n")}</code>
        </pre>
      </div>
    </div>
  );
}

/** Per-run detail table; the summary page shows every run, others a subset. */
export function RunsTable({ rows, results }: { rows: ExperimentRow[]; results: Results }) {
  const showTraffic = rows.some((r) => r.communication_mb);
  const showEpsilon = rows.some((r) => r.epsilon != null);
  return (
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
            {showTraffic && (
              <th scope="col" className="num">
                Traffic
              </th>
            )}
            {showEpsilon && (
              <th scope="col" className="num">
                ε
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.run}>
              <th scope="row">{r.setting}</th>
              <td className="num">{fmtAuroc(r.macro_auroc)}</td>
              <td className="num">{results.gap(r)}</td>
              <td className="num">{r.macro_f1 != null ? r.macro_f1.toFixed(3) : "–"}</td>
              <td className="num">
                {r.best_step != null
                  ? `${r.algorithm === "centralized" ? "epoch" : "round"} ${r.best_step} of ${r.steps_run}`
                  : "–"}
              </td>
              {showTraffic && (
                <td className="num">
                  {r.communication_mb ? `${formatInt(Math.round(r.communication_mb))} MB` : "–"}
                </td>
              )}
              {showEpsilon && <td className="num">{r.epsilon != null ? r.epsilon.toFixed(2) : "–"}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export const runCount = (n: number) => `${n} ${n === 1 ? "run" : "runs"}`;

/** Link to the next experiment page: the phases are a sequence. */
export function NextPage({ href, title, question }: { href: string; title: string; question: string }) {
  return (
    <nav className="next-page" aria-label="Next">
      <a href={href}>
        <span className="next-label">Next</span>
        <strong>{title}</strong>
        <span>{question}</span>
      </a>
    </nav>
  );
}
