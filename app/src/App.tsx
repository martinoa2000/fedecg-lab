import { useEffect, useState } from "react";
import { TooltipProvider } from "./components/charts";
import { useAppData, type AppData } from "./data";
import { Dataset } from "./views/Dataset";
import { Experiments } from "./views/Experiments";
import { Overview } from "./views/Overview";
import { Signals } from "./views/Signals";

const ROUTES = [
  { path: "", label: "Overview", view: Overview },
  { path: "dataset", label: "Dataset", view: Dataset },
  { path: "signals", label: "Signals", view: Signals },
  { path: "experiments", label: "Experiments", view: Experiments },
] as const;

// Mirrors the roadmap in README.md. Update when a phase lands.
const ROADMAP = [
  "Setup",
  "Exploration and preprocessing",
  "Centralised baseline",
  "Federated, IID",
  "Federated, non-IID",
  "Differential privacy",
  "Explainability",
  "Write-up",
];
const PHASES_DONE = 7;

function useRoute() {
  const read = () => window.location.hash.replace(/^#\/?/, "");
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => {
      setRoute(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="6" fill="var(--ink)" />
      <path
        d="M3 18h7l2-6 3 12 3-16 2 10h9"
        fill="none"
        stroke="var(--grid-minor)"
        strokeWidth="2.2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Shell({ data }: { data: AppData }) {
  const route = useRoute();
  const current = ROUTES.find((r) => r.path === route) ?? ROUTES[0];
  const View = current.view;

  useEffect(() => {
    document.title = current.path ? `${current.label}, fedecg-lab` : "fedecg-lab";
  }, [current]);

  return (
    <div className="shell">
      <header className="rail">
        <a className="brand" href="#/">
          <Logo />
          <span>fedecg-lab</span>
        </a>
        <nav className="nav" aria-label="Sections">
          {ROUTES.map((r) => (
            <a key={r.path} href={`#/${r.path}`} aria-current={r === current ? "page" : undefined}>
              {r.label}
            </a>
          ))}
        </nav>
        <div className="roadmap">
          <h2>Roadmap</h2>
          <ol>
            {ROADMAP.map((name, i) => (
              <li
                key={name}
                data-state={i < PHASES_DONE ? "done" : i === PHASES_DONE ? "current" : "todo"}
                aria-label={`Phase ${i + 1}, ${name}, ${i < PHASES_DONE ? "done" : i === PHASES_DONE ? "next" : "planned"}`}
              >
                <span className="step">
                  <span>{i + 1}</span>
                </span>
                <span>{name}</span>
              </li>
            ))}
          </ol>
        </div>
      </header>
      <main className="main">
        <View data={data} />
      </main>
    </div>
  );
}

export function App() {
  const state = useAppData();

  if (state.status === "loading") {
    return <div className="state" aria-busy="true" />;
  }

  if (state.status === "missing") {
    return (
      <div className="state">
        <div className="state-box">
          <h1 style={{ fontSize: "1.75rem" }}>No dashboard data yet</h1>
          <p className="lede">
            The dashboard reads files exported from PTB-XL. Run this from the repository root, then reload the
            page:
          </p>
          <pre>
            <code>uv run python scripts/export_dashboard.py</code>
          </pre>
          <p className="lede">
            If PTB-XL isn't downloaded yet, run <code>uv run python scripts/download_data.py</code> first.
          </p>
        </div>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <Shell data={state.data} />
    </TooltipProvider>
  );
}
