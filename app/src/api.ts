/**
 * Client for the local training server (scripts/serve.py), reached through
 * the Vite dev server's /api proxy. Every call rejects when the server is not
 * running, which the Train page treats as "offline", not as an error.
 */

export type OptionSpec =
  | { type: "choice"; values: (number | string | null)[]; default: number | string | null }
  | { type: "int" | "float"; min: number; max: number; default: number };

export type TrainOptions = {
  base_channels: number;
  crop_samples: number | null;
  epochs: number;
  learning_rate: number;
  loss: "bce" | "weighted_bce" | "focal";
  seeds: number;
  amplitude: number;
  noise: number;
  wander: number;
  lead_dropout: number;
};

export interface ExperimentConfig {
  name: string;
  phase: number | null;
  setting: string;
  script: "train_centralized" | "train_federated";
}

export type RunStatus = "queued" | "running" | "stopping" | "done" | "failed" | "stopped";

export interface Run {
  id: string;
  kind: "tune" | "experiment";
  label: string;
  config: string;
  options: Partial<TrainOptions>;
  status: RunStatus;
  created: number;
  started: number | null;
  finished: number | null;
  returncode: number | null;
  member: number;
  members: number;
  progress: { member: number; step: number; val_macro_auroc: number }[];
  result: number | null;
  log?: string[];
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error ?? `${response.status}`);
  return body as T;
}

export const api = {
  health: () => call<{ ok: boolean; busy: boolean }>("/health"),
  options: () => call<Record<keyof TrainOptions, OptionSpec>>("/options"),
  experiments: () => call<ExperimentConfig[]>("/experiments"),
  runs: () => call<Run[]>("/runs"),
  run: (id: string) => call<Run>(`/runs/${id}`),
  tune: (options: TrainOptions) => call<Run>("/tune", { method: "POST", body: JSON.stringify({ options }) }),
  experiment: (config: string) =>
    call<Run>("/experiments", { method: "POST", body: JSON.stringify({ config }) }),
  stop: (id: string) => call<Run>(`/runs/${id}/stop`, { method: "POST" }),
};
