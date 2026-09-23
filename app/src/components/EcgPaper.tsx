import { useId } from "react";

/**
 * A 12-lead ECG drawn at clinical scale on millimetre paper.
 *
 * Units inside the SVG are millimetres: 25 mm per second horizontally and
 * 10 mm per millivolt vertically, the standard paper speed and gain. Each
 * major square is therefore 0.2 s by 0.5 mV, exactly as on a printout.
 */

const MM_PER_S = 25;
const MM_PER_MV = 10;
const CAL_WIDTH = 8; // room for the 1 mV calibration pulse at the start of each row

// Standard 3 x 4 printout: each column shows 2.5 s of four leads.
const STANDARD_ROWS = [
  ["I", "aVR", "V1", "V4"],
  ["II", "aVL", "V2", "V5"],
  ["III", "aVF", "V3", "V6"],
];

export type EcgLayout = "standard" | "stacked";

interface Props {
  /** Microvolts, `[lead][sample]`. */
  signal: number[][];
  leads: string[];
  fs: number;
  layout?: EcgLayout;
  rhythmLead?: string;
  /** Draw the trace in once, as a recorder would. */
  animate?: boolean;
  label: string;
}

function tracePath(
  samples: number[],
  from: number,
  to: number,
  fs: number,
  x0: number,
  baseline: number,
): string {
  let d = "";
  for (let i = from; i < to; i++) {
    const x = x0 + ((i - from) / fs) * MM_PER_S;
    const y = baseline - (samples[i] / 1000) * MM_PER_MV;
    d += `${i === from ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
  }
  return d;
}

function calibration(x: number, baseline: number): string {
  const h = MM_PER_MV;
  return `M${x + 1} ${baseline}h1.5v${-h}h5v${h}h1.5`;
}

export function EcgPaper({
  signal,
  leads,
  fs,
  layout = "standard",
  rhythmLead = "II",
  animate = false,
  label,
}: Props) {
  const id = useId().replace(/:/g, "");
  const nSamples = signal[0]?.length ?? 0;
  const seconds = nSamples / fs;
  const stripWidth = seconds * MM_PER_S;
  const width = CAL_WIDTH + stripWidth;
  const leadIndex = (name: string) => leads.indexOf(name);

  type Segment = { lead: string; d: string; labelX: number; labelY: number; order: number };
  const segments: Segment[] = [];
  const cals: string[] = [];
  const dividers: number[][] = [];
  let height: number;

  if (layout === "standard") {
    const rowH = 30;
    const colSamples = Math.floor(nSamples / 4);
    const colWidth = stripWidth / 4;
    STANDARD_ROWS.forEach((row, r) => {
      const baseline = rowH * r + rowH * 0.6;
      cals.push(calibration(0, baseline));
      row.forEach((lead, c) => {
        const from = c * colSamples;
        const x0 = CAL_WIDTH + c * colWidth;
        segments.push({
          lead,
          d: tracePath(signal[leadIndex(lead)], from, from + colSamples, fs, x0, baseline),
          labelX: x0 + 1.2,
          labelY: rowH * r + 6,
          order: c,
        });
        if (c > 0) dividers.push([x0, baseline - 3, x0, baseline + 3]);
      });
    });
    const rhythmBaseline = rowH * 3 + rowH * 0.6;
    cals.push(calibration(0, rhythmBaseline));
    segments.push({
      lead: rhythmLead,
      d: tracePath(signal[leadIndex(rhythmLead)], 0, nSamples, fs, CAL_WIDTH, rhythmBaseline),
      labelX: CAL_WIDTH + 1.2,
      labelY: rowH * 3 + 6,
      order: 0,
    });
    height = rowH * 4 + 2;
  } else {
    const rowH = 20;
    leads.forEach((lead, r) => {
      const baseline = rowH * r + rowH * 0.6;
      cals.push(calibration(0, baseline));
      segments.push({
        lead,
        d: tracePath(signal[r], 0, nSamples, fs, CAL_WIDTH, baseline),
        labelX: CAL_WIDTH + 1.2,
        labelY: rowH * r + 5,
        order: 0,
      });
    });
    height = rowH * leads.length + 2;
  }

  return (
    <svg
      className="ecg-paper"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
      data-animate={animate || undefined}
    >
      <defs>
        <pattern id={`minor-${id}`} width="1" height="1" patternUnits="userSpaceOnUse">
          <path d="M1 0V1H0" className="ecg-grid-minor" />
        </pattern>
        <pattern id={`major-${id}`} width="5" height="5" patternUnits="userSpaceOnUse">
          <rect width="5" height="5" fill={`url(#minor-${id})`} />
          <path d="M5 0V5H0" className="ecg-grid-major" />
        </pattern>
      </defs>
      <rect width={width} height={height} className="ecg-paper-bg" />
      <rect width={width} height={height} fill={`url(#major-${id})`} />
      <g className="ecg-cal">
        {cals.map((d, i) => (
          <path key={i} d={d} />
        ))}
      </g>
      <g className="ecg-dividers">
        {dividers.map(([x1, y1, x2, y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />
        ))}
      </g>
      <g className="ecg-trace">
        {segments.map((s) => (
          <path
            key={`${s.lead}-${s.labelY}`}
            d={s.d}
            pathLength={1}
            style={{ "--order": s.order } as React.CSSProperties}
          />
        ))}
      </g>
      <g className="ecg-labels">
        {segments.map((s) => (
          <text key={`${s.lead}-${s.labelY}-label`} x={s.labelX} y={s.labelY}>
            {s.lead}
          </text>
        ))}
      </g>
    </svg>
  );
}
