import { Fragment, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { useTip } from "./charts";

/* ------------------------------------------------------------ AUROC scale */

/**
 * A zoomed AUROC axis shared by every chart on the page. Differences between
 * runs are a few thousandths, so a 0-1 axis would hide them; every chart that
 * compares runs uses this same domain, so a gap means the same thing everywhere.
 */
export function aurocDomain(values: number[], step = 0.01): [number, number] {
  const lo = Math.floor((Math.min(...values) - step / 2) / step) * step;
  const hi = Math.ceil((Math.max(...values) + step / 4) / step) * step;
  return [lo, hi];
}

export function ticksOf([lo, hi]: [number, number], step = 0.01): number[] {
  const n = Math.round((hi - lo) / step);
  const every = n > 8 ? 2 : 1;
  return Array.from({ length: n + 1 }, (_, i) => lo + i * step).filter((_, i) => i % every === 0);
}

export const fmtAuroc = (v: number) => v.toFixed(3);

export const fmtDelta = (v: number) => {
  // Round half away from zero (Math.round alone rounds -0.0155 to -0.015).
  // The epsilon absorbs float error: 0.9013 - 0.9168 is -0.015499999...
  const r = (Math.sign(v) * Math.round(Math.abs(v) * 1000 + 1e-9)) / 1000;
  if (r === 0) return "±0.000";
  return `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(3)}`;
};

/* --------------------------------------------------------------- dot plot */

export interface DotPoint {
  key: string;
  value: number;
  /** A CSS color for the dot. */
  color: string;
  /** Hollow dots mark the first of a pair (the "before" of a dumbbell). */
  hollow?: boolean;
  tip: ReactNode;
  label: string;
}

export interface DotRow {
  key: string;
  /** Rows sharing a group get one heading above the first of them. */
  group?: string;
  label: ReactNode;
  points: DotPoint[];
  /** Right-hand column, e.g. the gap to the baseline. */
  value: ReactNode;
}

export interface Reference {
  key: string;
  value: number;
  label: string;
}

/**
 * Rows of dots on one horizontal AUROC axis, with vertical reference lines.
 * Two points in a row are joined, which makes it a dumbbell.
 */
export function DotPlot({
  rows,
  references,
  domain,
  ariaLabel,
  tickStep = 0.01,
  format = fmtAuroc,
  metric = "macro AUROC",
}: {
  rows: DotRow[];
  references: Reference[];
  domain: [number, number];
  ariaLabel: string;
  /** Spacing of axis ticks, in data units. */
  tickStep?: number;
  /** Formats a value for screen readers. */
  format?: (v: number) => string;
  /** What the values are, for screen readers. */
  metric?: string;
}) {
  const tip = useTip();
  const [lo, hi] = domain;
  const pct = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  // Reference labels closer than an eighth of the axis would overlap: stack them.
  const sortedRefs = [...references].sort((a, b) => a.value - b.value);
  const stacked = sortedRefs.some((r, i) => i > 0 && (r.value - sortedRefs[i - 1].value) / (hi - lo) < 0.12);
  return (
    <div className="dotplot" role="group" aria-label={ariaLabel}>
      <div className="dot-row dot-refs" data-stacked={stacked || undefined} aria-hidden>
        <span />
        <span className="dot-track">
          {references.map((r, i) => (
            <span
              key={r.key}
              className="ref-label"
              style={{ left: pct(r.value), bottom: stacked && i % 2 ? 20 : 4 }}
            >
              {r.label}
            </span>
          ))}
        </span>
        <span />
      </div>
      {rows.map((row, i) => {
        const values = row.points.map((p) => p.value);
        const from = Math.min(...values);
        const to = Math.max(...values);
        const heading = row.group && row.group !== rows[i - 1]?.group ? row.group : null;
        return (
          <Fragment key={row.key}>
          {heading && <p className="dot-group">{heading}</p>}
          <div className="dot-row" role="list" aria-label={typeof row.label === "string" ? row.label : undefined}>
            <span className="name">{row.label}</span>
            <span className="dot-track">
              {references.map((r) => (
                <i key={r.key} className="ref-line" style={{ left: pct(r.value) }} aria-hidden />
              ))}
              {row.points.length > 1 && (
                <i className="dot-link" style={{ left: pct(from), width: `calc(${pct(to)} - ${pct(from)})` }} aria-hidden />
              )}
              {row.points.map((p) => (
                <span
                  key={p.key}
                  role="listitem"
                  className="dot-hit"
                  data-hollow={p.hollow || undefined}
                  style={{ left: pct(p.value), "--c": p.color } as React.CSSProperties}
                  aria-label={`${p.label}: ${metric} ${format(p.value)}`}
                  {...tip(p.tip)}
                >
                  <i className="dot" />
                </span>
              ))}
            </span>
            <span className="value">{row.value}</span>
          </div>
          </Fragment>
        );
      })}
      <div className="dot-row dot-axis" aria-hidden>
        <span />
        <span className="dot-track">
          {ticksOf(domain, tickStep).map((t) => (
            <span key={t} className="tick" style={{ left: pct(t) }}>
              {t.toFixed(tickStep >= 0.5 ? 1 : tickStep >= 0.01 ? 2 : 3)}
            </span>
          ))}
        </span>
        <span />
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- curve chart */

export interface Series {
  key: string;
  label: string;
  color: string;
  points: { x: number; y: number }[];
  /** The selected step, marked with a dot. */
  best?: number;
}

export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(640);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

/**
 * Validation macro AUROC per epoch or round, one line per run, with a
 * crosshair that reads every series at the hovered step.
 */
export function CurveChart({
  series,
  xLabel,
  ariaLabel,
  height = 236,
}: {
  series: Series[];
  xLabel: string;
  ariaLabel: string;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { top: 12, right: 16, bottom: 44, left: 44 };
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const maxX = Math.max(...xs);
  // Early epochs sit far below the plateau; start the axis near the pack.
  const sorted = [...ys].sort((a, b) => a - b);
  const floor = sorted[Math.floor(sorted.length * 0.1)] ?? Math.min(...ys);
  const [lo, hi] = aurocDomain([floor, Math.max(...ys)], 0.02);
  const plotW = Math.max(width - pad.left - pad.right, 10);
  const plotH = height - pad.top - pad.bottom;
  const x = (v: number) => pad.left + ((v - 1) / Math.max(maxX - 1, 1)) * plotW;
  const y = (v: number) => pad.top + (1 - (Math.max(v, lo) - lo) / (hi - lo)) * plotH;
  const path = (pts: Series["points"]) =>
    pts.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join("");

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const step = Math.round(1 + ((e.clientX - box.left) / box.width) * (maxX - 1));
    setHover(Math.min(Math.max(step, 1), maxX));
  };

  const readout = hover
    ? series
        .map((s) => ({ s, p: s.points.find((p) => p.x === hover) }))
        .filter((r): r is { s: Series; p: { x: number; y: number } } => Boolean(r.p))
        .sort((a, b) => b.p.y - a.p.y)
    : [];

  // Keyboard: focus lands on the selected step; arrows move the crosshair.
  const onKeyDown = (e: React.KeyboardEvent) => {
    const delta = { ArrowLeft: -1, ArrowRight: 1, Home: -maxX, End: maxX }[e.key];
    if (delta === undefined) return;
    e.preventDefault();
    setHover((h) => Math.min(Math.max((h ?? 1) + delta, 1), maxX));
  };

  return (
    <figure
      className="curve"
      ref={ref}
      style={{ margin: 0 }}
      tabIndex={0}
      aria-label={`${ariaLabel}. Use the arrow keys to read values.`}
      onKeyDown={onKeyDown}
      onFocus={() => setHover(series[0]?.best ?? maxX)}
      onBlur={() => setHover(null)}
    >
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        {ticksOf([lo, hi], 0.02).map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.left} x2={pad.left + plotW} y1={y(t)} y2={y(t)} />
            <text className="axis-text" x={pad.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {[1, ...Array.from({ length: 5 }, (_, i) => Math.round(((i + 1) * maxX) / 5))]
          .filter((v, i, a) => a.indexOf(v) === i)
          .map((t) => (
            <text key={t} className="axis-text" x={x(t)} y={pad.top + plotH + 18} textAnchor="middle">
              {t}
            </text>
          ))}
        <text className="axis-text" x={pad.left + plotW / 2} y={height - 4} textAnchor="middle">
          {xLabel}
        </text>
        {hover && <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={pad.top} y2={pad.top + plotH} />}
        {series.map((s) => (
          <path key={s.key} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {series.map((s) => {
          const best = s.points.find((p) => p.x === s.best);
          return best ? (
            <circle key={s.key} cx={x(best.x)} cy={y(best.y)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
          ) : null;
        })}
        <rect
          x={pad.left}
          y={pad.top}
          width={plotW}
          height={plotH}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => setHover(null)}
        />
      </svg>
      {hover && readout.length > 0 && (
        <div className="curve-tip" style={{ left: Math.min(x(hover) + 12, width - 200) }} aria-live="polite">
          <span className="curve-tip-head">
            {xLabel} {hover}
          </span>
          {readout.map(({ s, p }) => (
            <span key={s.key} className="curve-tip-row">
              <i style={{ background: s.color }} />
              <strong>{fmtAuroc(p.y)}</strong>
              <span>{s.label}</span>
            </span>
          ))}
        </div>
      )}
      {series.length > 1 && (
        <figcaption className="legend" style={{ marginTop: 8 }}>
          {series.map((s) => (
            <span key={s.key}>
              <i className="line-key" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}

/* ---------------------------------------------------------- epsilon chart */

export interface EpsilonPoint {
  /** Privacy budget; null is the same recipe trained without privacy. */
  epsilon: number | null;
  value: number;
  tip: ReactNode;
}

export interface EpsilonSeries {
  key: string;
  label: string;
  color: string;
  points: EpsilonPoint[];
}

/**
 * Test AUROC against the privacy budget. Epsilon values sit at even steps
 * (1, 3, 8, no privacy): they are the settings that were run, not a
 * continuous scale, and spacing them evenly keeps the strictest budgets
 * readable.
 */
export function EpsilonChart({
  series,
  references,
  ariaLabel,
  height = 260,
}: {
  series: EpsilonSeries[];
  references: Reference[];
  ariaLabel: string;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTip();
  const budgets = [...new Set(series.flatMap((s) => s.points.map((p) => p.epsilon)))].sort(
    (a, b) => (a ?? Infinity) - (b ?? Infinity),
  );
  const values = [...series.flatMap((s) => s.points.map((p) => p.value)), ...references.map((r) => r.value)];
  const [lo, hi] = aurocDomain(values, 0.02);
  const labelRoom = Math.min(150, width * 0.3);
  const pad = { top: 16, right: labelRoom, bottom: 44, left: 44 };
  const plotW = Math.max(width - pad.left - pad.right, 10);
  const plotH = height - pad.top - pad.bottom;
  const x = (eps: number | null) =>
    pad.left + (budgets.length > 1 ? (budgets.indexOf(eps) / (budgets.length - 1)) * plotW : plotW / 2);
  const y = (v: number) => pad.top + (1 - (v - lo) / (hi - lo)) * plotH;
  const fmtEps = (eps: number | null) => (eps === null ? "No privacy" : `ε = ${eps}`);

  return (
    <figure className="curve" ref={ref} style={{ margin: 0 }} aria-label={ariaLabel}>
      <svg width={width} height={height} role="img" aria-label={ariaLabel}>
        {ticksOf([lo, hi], 0.02).map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.left} x2={pad.left + plotW} y1={y(t)} y2={y(t)} />
            <text className="axis-text" x={pad.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {references.map((r) => (
          <g key={r.key}>
            <line className="ref" x1={pad.left} x2={pad.left + plotW} y1={y(r.value)} y2={y(r.value)} />
            <text className="axis-text" x={pad.left + plotW + 8} y={y(r.value)} dy="0.32em">
              {r.label}
            </text>
          </g>
        ))}
        {budgets.map((eps) => (
          <text key={String(eps)} className="axis-text" x={x(eps)} y={pad.top + plotH + 18} textAnchor="middle">
            {eps === null ? "none" : eps}
          </text>
        ))}
        <text className="axis-text" x={pad.left + plotW / 2} y={height - 4} textAnchor="middle">
          Privacy budget ε (smaller is more private)
        </text>
        {series.map((s) => {
          const pts = [...s.points].sort((a, b) => budgets.indexOf(a.epsilon) - budgets.indexOf(b.epsilon));
          const last = pts[pts.length - 1];
          return (
            <g key={s.key}>
              <path
                d={pts.map((p, i) => `${i ? "L" : "M"}${x(p.epsilon).toFixed(1)},${y(p.value).toFixed(1)}`).join("")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
              />
              {last && (
                <text className="series-label" x={x(last.epsilon) + 10} y={y(last.value)} dy="0.32em">
                  {s.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {/* Dots are HTML so they get the shared tooltip and keyboard focus. */}
      <div className="eps-dots">
        {series.flatMap((s) =>
          s.points.map((p) => (
            <span
              key={`${s.key}-${p.epsilon}`}
              className="dot-hit"
              role="img"
              style={{ left: x(p.epsilon), top: y(p.value), "--c": s.color } as React.CSSProperties}
              aria-label={`${s.label}, ${fmtEps(p.epsilon)}: macro AUROC ${fmtAuroc(p.value)}`}
              {...tip(p.tip)}
            >
              <i className="dot" />
            </span>
          )),
        )}
      </div>
    </figure>
  );
}
