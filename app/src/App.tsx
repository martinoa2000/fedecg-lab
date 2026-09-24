import { useEffect, useState, type ReactElement } from "react";
import { TooltipProvider } from "./components/charts";
import { useAppData, type AppData } from "./data";
import { Dataset } from "./views/Dataset";
import { Baseline } from "./views/experiments/Baseline";
import { Explainability } from "./views/experiments/Explainability";
import { Federated } from "./views/experiments/Federated";
import { Privacy } from "./views/experiments/Privacy";
import { Results } from "./views/experiments/Results";
import { Train } from "./views/experiments/Train";
import { Overview } from "./views/Overview";
import { Signals } from "./views/Signals";

interface Route {
  path: string;
  label: string;
  view: (props: { data: AppData }) => ReactElement;
}

const NAV: { label: string; routes: Route[] }[] = [
  {
    label: "Data",
    routes: [
      { path: "", label: "Overview", view: Overview },
      { path: "dataset", label: "Dataset", view: Dataset },
      { path: "signals", label: "Signals", view: Signals },
    ],
  },
  {
    label: "Experiments",
    routes: [
      { path: "results", label: "Results", view: Results },
      { path: "baseline", label: "Baseline", view: Baseline },
      { path: "federated", label: "Federated", view: Federated },
      { path: "privacy", label: "Privacy", view: Privacy },
      { path: "explainability", label: "Explainability", view: Explainability },
      { path: "train", label: "Tuning and training", view: Train },
    ],
  },
];

const ROUTES = NAV.flatMap((group) => group.routes);

/** Old links keep working: the single Experiments page became Results. */
const ALIASES: Record<string, string> = { experiments: "results" };

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
const PHASES_DONE = 8;

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
  const path = ALIASES[route] ?? route;
  const current = ROUTES.find((r) => r.path === path) ?? ROUTES[0];
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
          {NAV.map((group) => (
            <div key={group.label} className="nav-group" role="group" aria-labelledby={`nav-${group.label}`}>
              <h2 id={`nav-${group.label}`} className="nav-heading">
                {group.label}
              </h2>
              {group.routes.map((r) => (
                <a key={r.path} href={`#/${r.path}`} aria-current={r === current ? "page" : undefined}>
                  {r.label}
                </a>
              ))}
            </div>
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
        <footer className="site-footer">
          <p>
            ECG data: PTB-XL (Wagner et al., 2020), version 1.0.3 on{" "}
            <a href="https://physionet.org/content/ptb-xl/1.0.3/">PhysioNet</a>, licensed under{" "}
            <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; recordings shown here are
            band-pass filtered and annotated. Research code, not a medical device.{" "}
            <a href="https://github.com/martinoa2000/fedecg-lab">Source on GitHub</a>.
          </p>
        </footer>
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
