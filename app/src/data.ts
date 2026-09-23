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
  setting: string;
  macro_auroc: number;
  epsilon?: number | null;
  [column: string]: unknown;
}

export interface AppData {
  dataset: Dataset;
  signals: Signals;
  experiments: ExperimentRow[];
}

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
      getJson<{ rows: ExperimentRow[] }>("experiments.json"),
    ])
      .then(([dataset, signals, experiments]) =>
        setState({ status: "ready", data: { dataset, signals, experiments: experiments.rows } }),
      )
      .catch(() => setState({ status: "missing" }));
  }, []);
  return state;
}

export const formatInt = (n: number) => n.toLocaleString("en-US");
export const formatPct = (n: number, digits = 1) => `${n.toFixed(digits)}%`;
