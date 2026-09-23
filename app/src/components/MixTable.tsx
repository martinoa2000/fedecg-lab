import { useState } from "react";
import { formatInt, formatPct, type MixRow, type Superclass } from "../data";
import { Swatch, useTip } from "./charts";

/** Records and label mix per group (site, device or simulated hospital). */
export function MixTable<T extends MixRow>({
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
