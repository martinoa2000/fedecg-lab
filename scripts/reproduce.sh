#!/usr/bin/env bash
# Reproduce every number in the README, from download to dashboard.
#
# Runs the phases in order; each step writes its tables to results/tables/ and
# one row per experiment to results/tables/experiments.csv. Total time on an
# Apple M-series laptop: about three and a half hours, most of it phase 6
# (DP-SGD computes a gradient per record) and the tuning study.
#
# Usage:
#   scripts/reproduce.sh            # everything
#   scripts/reproduce.sh 5 6        # only phases 5 and 6 (needs earlier checkpoints)
#   scripts/reproduce.sh tune       # only the baseline tuning study (about 50 minutes)

set -euo pipefail
cd "$(dirname "$0")/.."

phases=("$@")
[[ ${#phases[@]} -eq 0 ]] && phases=(2 3 tune 4 5 6 7)
wants() { [[ " ${phases[*]} " == *" $1 "* ]]; }
run() { echo "+ $*" >&2; uv run python "$@"; }

run scripts/download_data.py

if wants 2; then
  run scripts/explore_data.py
fi

if wants 3; then
  for config in centralized centralized_untuned centralized_plain centralized_ensemble; do
    run scripts/train_centralized.py --config "$config.yaml"
  done
fi

if wants tune; then
  # Validation-only study of training options; its winner is centralized.yaml.
  for config in tune_baseline tune_wide tune_augment tune_weighted tune_focal tune_ensemble tune_wide_augment tune_best; do
    run scripts/tune.py --config "$config.yaml"
  done
fi

if wants 4; then
  for n in 5 10 20; do
    run scripts/train_federated.py --config "fedavg_iid_$n.yaml"
  done
fi

if wants 5; then
  for partition in site device dirichlet; do
    for strategy in fedavg fedprox; do
      run scripts/train_federated.py --config "${strategy}_$partition.yaml"
    done
  done
fi

if wants 6; then
  for config in dp_none dp_eps8 dp_eps3 dp_eps1; do
    run scripts/train_centralized.py --config "$config.yaml"
  done
  for config in fed_dp_none fed_dp_eps8 fed_dp_eps3 fed_dp_eps1; do
    run scripts/train_federated.py --config "$config.yaml"
  done
fi

if wants 7; then
  # Explains checkpoints/centralized.pt, written in phase 3.
  run scripts/explain_model.py
fi

run scripts/export_dashboard.py
echo "Done. Results in results/tables/; dashboard: npm --prefix app run dev" >&2
