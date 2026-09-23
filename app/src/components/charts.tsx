import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { CLASS_INFO, formatInt, type Superclass } from "../data";

/* ------------------------------------------------------------- tooltip */

type Tip = { x: number; y: number; content: ReactNode } | null;
const TipContext = createContext<(tip: Tip) => void>(() => {});

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip>(null);
  return (
    <TipContext.Provider value={setTip}>
      {children}
      {tip && (
        <div className="tooltip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.content}
        </div>
      )}
    </TipContext.Provider>
  );
}

/** Props that show `content` above an element on hover and keyboard focus. */
export function useTip() {
  const setTip = useContext(TipContext);
  return useCallback(
    (content: ReactNode) => {
      const show = (el: Element) => {
        const r = el.getBoundingClientRect();
        setTip({ x: r.left + r.width / 2, y: r.top, content });
      };
      return {
        tabIndex: 0,
        onPointerEnter: (e: React.PointerEvent) => show(e.currentTarget),
        onPointerLeave: () => setTip(null),
        onFocus: (e: React.FocusEvent) => show(e.currentTarget),
        onBlur: () => setTip(null),
      };
    },
    [setTip],
  );
}

/* ---------------------------------------------------------- class swatch */

export function Swatch({ cls }: { cls: Superclass }) {
  return <span className="swatch" style={{ "--c": `var(--${cls})` } as React.CSSProperties} aria-hidden />;
}

export function ClassName({ cls, long = false }: { cls: Superclass; long?: boolean }) {
  return (
    <span className="label-chip">
      <Swatch cls={cls} />
      {cls}
      {long && <span style={{ color: "var(--ink-2)", fontWeight: 400 }}>{CLASS_INFO[cls].name}</span>}
    </span>
  );
}

/* ------------------------------------------------------ horizontal bars */

export interface BarDatum {
  key: string;
  label: ReactNode;
  value: number;
  display: string;
  color?: string;
  tip: ReactNode;
  ticks?: number[];
}

export function Bars({ data, max, ariaLabel }: { data: BarDatum[]; max?: number; ariaLabel: string }) {
  const tip = useTip();
  const top = max ?? Math.max(...data.map((d) => d.value));
  return (
    <div className="bars" role="list" aria-label={ariaLabel}>
      {data.map((d) => (
        <div key={d.key} className="bar-row" role="listitem" {...tip(d.tip)}>
          <span className="name">{d.label}</span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ width: `${(d.value / top) * 100}%`, "--c": d.color } as React.CSSProperties}
              aria-hidden
            />
            {d.ticks && (
              <span className="split-ticks" aria-hidden>
                {d.ticks.map((t, i) => (
                  <i key={i} style={{ left: `${(t / top) * 100}%` }} />
                ))}
              </span>
            )}
          </span>
          <span className="value">{d.display}</span>
        </div>
      ))}
    </div>
  );
}

export function CountTip({ title, n, total, note }: { title: string; n: number; total: number; note?: string }) {
  return (
    <>
      <strong>{title}</strong>
      <br />
      {formatInt(n)} records, {((n / total) * 100).toFixed(1)}% of {formatInt(total)}
      {note && (
        <>
          <br />
          {note}
        </>
      )}
    </>
  );
}
