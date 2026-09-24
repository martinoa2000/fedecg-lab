import type { SaliencyExample } from "../data";
import { useWidth } from "./results";

/**
 * One lead of an example record: the trace, the ST segments shaded behind it,
 * and the attribution as a band of bars below it. Attribution is a magnitude,
 * so it is one hue whose opacity follows the value.
 */
export function SaliencyStrip({
  example,
  lead,
  leadName,
  fs,
  height = 132,
}: {
  example: SaliencyExample;
  lead: number;
  leadName: string;
  fs: number;
  height?: number;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const signal = example.signal[lead];
  const attribution = example.attribution[lead];
  const n = signal.length;
  const band = 26;
  const pad = { top: 8, bottom: band + 22, left: 40, right: 8 };
  const plotW = Math.max(width - pad.left - pad.right, 10);
  const plotH = height - pad.top - pad.bottom;
  const lo = Math.min(...signal);
  const hi = Math.max(...signal);
  const x = (i: number) => pad.left + (i / (n - 1)) * plotW;
  const y = (v: number) => pad.top + (1 - (v - lo) / Math.max(hi - lo, 1)) * plotH;
  const path = signal.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const bandTop = pad.top + plotH + 6;
  const barW = Math.max(plotW / n, 1);
  const seconds = Math.floor(n / fs);

  return (
    <div className="strip" ref={ref}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Lead ${leadName}: ECG trace with ST segments shaded and attribution shown beneath`}
      >
        {example.segments.st.map(([a, b]) => (
          <rect key={a} className="st-span" x={x(a)} y={pad.top} width={Math.max(x(b) - x(a), 1)} height={plotH} />
        ))}
        <path d={path} className="trace" />
        <text className="axis-text" x={pad.left - 8} y={pad.top + plotH / 2} dy="0.32em" textAnchor="end">
          {leadName}
        </text>
        {attribution.map((a, i) =>
          a >= 4 ? (
            <rect
              key={i}
              className="attr-bar"
              x={x(i) - barW / 2}
              y={bandTop + band * (1 - a / 100)}
              width={barW}
              height={(band * a) / 100}
              opacity={0.25 + (0.75 * a) / 100}
            />
          ) : null,
        )}
        <line className="grid" x1={pad.left} x2={pad.left + plotW} y1={bandTop + band} y2={bandTop + band} />
        {Array.from({ length: seconds + 1 }, (_, s) => (
          <text
            key={s}
            className="axis-text"
            x={x(Math.min(s * fs, n - 1))}
            y={height - 4}
            // The last tick sits on the right edge; anchor it there so it is not clipped.
            textAnchor={s === seconds ? "end" : s === 0 ? "start" : "middle"}
          >
            {s} s
          </text>
        ))}
      </svg>
    </div>
  );
}
