import { useEffect, useState } from "react";

export type Superclass = "NORM" | "MI" | "STTC" | "CD" | "HYP";

export const CLASS_INFO: Record<Superclass, { name: string; blurb: string }> = {
  NORM: { name: "Normal", blurb: "No diagnostic abnormality." },
  MI: { name: "Myocardial infarction", blurb: "Heart muscle damaged by blocked blood flow." },
  STTC: { name: "ST/T change", blurb: "Altered repolarisation in the ST segment or T wave." },
  CD: { name: "Conduction disturbance", blurb: "The electrical impulse travels abnormally." },
  HYP: { name: "Hypertrophy", blurb: "Thickened heart wall or enlarged chamber." },
};

export interface MixRow {
  n_records: number;
  NORM: number;
  MI: number;
  STTC: number;
  CD: number;
  HYP: number;
}

export interface Dataset {
  superclasses: Superclass[];
  records_total: number;
  records_labeled: number;
  patients: number;
  split: { train: number; val: number; test: number };
  split_folds: { train: number[]; val: number; test: number };
  sampling_rate_hz: number;
  bandpass_hz: [number, number];
  distribution: {
    superclass: Superclass;
    all_n: number;
    all_pct: number;
    train_pct: number;
    val_pct: number;
    test_pct: number;
  }[];
  cooccurrence: number[][];
  cardinality: { n_labels: number; n_records: number }[];
  combinations: { labels: string; n_records: number }[];
  by_site: (MixRow & { site: number })[];
  by_device: (MixRow & { device: string })[];
  age: { median: number; histogram: number[]; bin_width: number };
  sex: { male: number; female: number };
}

export interface EcgRecord {
  ecg_id: number;
  labels: Superclass[];
  age: number | null;
  sex: "male" | "female";
  device: string;
  site: number | null;
  report: string;
  scp_codes: Record<string, number>;
  validated: boolean;
  strat_fold: number;
  /** Microvolts, `[lead][sample]`. */
  raw: number[][];
  filtered: number[][];
}

export interface Signals {
  fs: number;
  leads: string[];
  examples: Record<Superclass, number[]>;
  records: EcgRecord[];
}

export interface ExperimentRow {
  phase: number;
  /** Config file stem; unique per row. */
  run: string;
  setting: string;
  algorithm?: "centralized" | "fedavg" | "fedprox" | string;
  partition?: "none" | "iid" | "site" | "device" | "dirichlet" | string;
  n_clients?: number | null;
  macro_auroc: number;
  macro_f1?: number | null;
  auroc_NORM?: number;
  auroc_MI?: number;
  auroc_STTC?: number;
  auroc_CD?: number;
  auroc_HYP?: number;
  /** Epoch (centralised) or round (federated) the model was selected at. */
  best_step?: number | null;
  steps_run?: number | null;
  communication_mb?: number | null;
  epsilon?: number | null;
  seconds?: number | null;
}

export interface CurvePoint {
  /** Epoch for centralised runs, round for federated ones. */
  step: number;
  train_loss: number;
  val_loss: number;
  val_macro_auroc: number;
}

export type ClientRow = MixRow & { client: string };

export interface PublishedResult {
  model: string;
  macro_auroc: number;
  source: string;
}

export interface Experiments {
  rows: ExperimentRow[];
  histories: Record<string, CurvePoint[]>;
  clients: Record<string, ClientRow[]>;
  published: PublishedResult[];
}

export type Segment = "qrs" | "st" | "t" | "other";

export interface SegmentRow {
  superclass: Superclass;
  method: "integrated_gradients" | "integrated_gradients_random" | "grad_cam" | "grad_cam_random";
  segment: Segment;
  n_records: number;
  attribution_share: number;
  time_share: number;
  enrichment: number | null;
}

export type LeadShareRow = { superclass: Superclass } & Record<string, number>;

export interface SanityRow {
  superclass: Superclass;
  n_records: number;
  integrated_gradients: number | null;
  grad_cam: number | null;
  /** Share of records whose random-weight Grad-CAM is not all zero. */
  grad_cam_random_nonzero?: number;
}

export interface SaliencyExample {
  superclass: Superclass;
  probability: number;
  /** Microvolts, `[lead][sample]`. */
  signal: number[][];
  /** |Integrated Gradients|, 0-100 scaled by the record's maximum, `[lead][sample]`. */
  attribution: number[][];
  /** `[start, end)` sample ranges per beat segment, from lead II. */
  segments: Record<"qrs" | "st" | "t", [number, number][]>;
}

export interface Explain {
  segments: SegmentRow[];
  leads: LeadShareRow[];
  sanity: SanityRow[];
  examples: SaliencyExample[];
  fs?: number;
  lead_names?: string[];
}

export interface AppData {
  dataset: Dataset;
  signals: Signals;
  experiments: Experiments;
  explain: Explain;
}

const EMPTY_EXPLAIN: Explain = { segments: [], leads: [], sanity: [], examples: [] };

export type LoadState =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "ready"; data: AppData };

async function getJson<T>(name: string): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  return (await response.json()) as T;
}

export function useAppData(): LoadState {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  useEffect(() => {
    Promise.all([
      getJson<Dataset>("dataset.json"),
      getJson<Signals>("signals.json"),
      getJson<Partial<Experiments>>("experiments.json"),
      // Optional: only present once scripts/explain_model.py has run.
      getJson<Explain>("explain.json").catch(() => EMPTY_EXPLAIN),
    ])
      .then(([dataset, signals, experiments, explain]) =>
        setState({
          status: "ready",
          data: {
            dataset,
            signals,
            // Older exports only carried `rows`.
            experiments: {
              rows: experiments.rows ?? [],
              histories: experiments.histories ?? {},
              clients: experiments.clients ?? {},
              published: experiments.published ?? [],
            },
            explain,
          },
        }),
      )
      .catch(() => setState({ status: "missing" }));
  }, []);
  return state;
}

export const formatInt = (n: number) => n.toLocaleString("en-US");
export const formatPct = (n: number, digits = 1) => `${n.toFixed(digits)}%`;
