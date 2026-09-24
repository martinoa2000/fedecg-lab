#!/usr/bin/env python
"""Where does the trained model look? Saliency on test ECGs (roadmap phase 7).

For each superclass, takes test records the model detects correctly (true
label present, probability above the class's validation-tuned threshold),
computes Integrated Gradients for that class, delineates the beats in lead II,
and measures how much attribution lands in the QRS complex, the ST segment and
the T wave, relative to the time each covers. A model-randomization check
repeats the attributions with random weights to show that the maps depend on
what the model learned.

Writes:

    results/tables/saliency_segments.csv   attribution vs. time share per class and segment
    results/tables/saliency_leads.csv      share of attribution per lead, per class
    results/tables/saliency_sanity.csv     median Spearman of trained vs. random-weight maps
    results/figures/saliency_<CLASS>.png   one example per class, IG over the signal
    results/saliency_examples.npz          example signals and maps for the dashboard
                                           (git-ignored: it contains PTB-XL waveforms)

Usage:
    python scripts/explain_model.py                   # explains checkpoints/centralized.pt
    python scripts/explain_model.py --per-class 50    # faster
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fedecg.data.constants import LEAD_NAMES, SUPERCLASSES
from fedecg.data.pipeline import prepare_data
from fedecg.explain.saliency import (
    SEGMENTS,
    beat_segments,
    grad_cam,
    integrated_gradients,
    lead_shares,
    randomization_similarity,
    segment_shares,
    summarize_enrichment,
)
from fedecg.models.resnet1d import build_model
from fedecg.paths import (
    CACHE_DIR,
    CHECKPOINT_DIR,
    FIGURES_DIR,
    PTBXL_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    ensure_dir,
)
from fedecg.seed import new_generator, set_seed
from fedecg.training.loop import make_loader, predict, resolve_device

LEAD_II = LEAD_NAMES.index("II")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "centralized.pt",
        help="Model to explain",
    )
    parser.add_argument("--root", type=Path, default=PTBXL_DIR, help="PTB-XL directory")
    parser.add_argument(
        "--cache", type=Path, default=CACHE_DIR / "ptbxl_100hz.npz", help="Waveform cache"
    )
    parser.add_argument("--no-cache", action="store_true", help="Always parse WFDB records")
    parser.add_argument("--per-class", type=int, default=100, help="Records explained per class")
    parser.add_argument("--steps", type=int, default=32, help="Integrated Gradients steps")
    parser.add_argument("--tables", type=Path, default=TABLES_DIR, help="Output for CSVs")
    parser.add_argument("--figures", type=Path, default=FIGURES_DIR, help="Output for figures")
    parser.add_argument(
        "--examples",
        type=Path,
        default=RESULTS_DIR / "saliency_examples.npz",
        help="Example maps for the dashboard",
    )
    return parser.parse_args(argv)


def plot_example(
    signal_mv: np.ndarray, attribution: np.ndarray, masks: dict, fs: float, title: str
) -> plt.Figure:
    """Lead II and the lead carrying most attribution, IG shaded, ST segments marked."""
    top = int(np.argmax(np.abs(attribution).sum(axis=1)))
    leads = [LEAD_II] if top == LEAD_II else [LEAD_II, top]
    time = np.arange(signal_mv.shape[1]) / fs
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 2.2 * len(leads)), sharex=True)
    axes = np.atleast_1d(axes)
    scale = np.abs(attribution).max() or 1.0
    for ax, lead in zip(axes, leads, strict=True):
        weight = np.abs(attribution[lead]) / scale
        low, high = signal_mv[lead].min(), signal_mv[lead].max()
        ax.fill_between(
            time, low, high, where=masks["st"], color="0.9", step="mid", label="ST segment"
        )
        ax.bar(
            time,
            (high - low) * weight,
            bottom=low,
            width=1 / fs,
            color="tab:red",
            alpha=0.45,
            label="|IG|",
        )
        ax.plot(time, signal_mv[lead], color="black", linewidth=0.8)
        ax.set_ylabel(f"{LEAD_NAMES[lead]} (mV)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    config = state["config"]
    seed = int(config["seed"])
    set_seed(seed)
    fs = config["data"]["sampling_rate_hz"]

    data = prepare_data(config, root=args.root, cache_path=None if args.no_cache else args.cache)
    device = resolve_device(config["training"].get("device", "auto"))
    model = build_model(config["model"]).to(device).eval()
    model.load_state_dict(state["model_state"])
    thresholds = np.asarray(state["thresholds"])

    prob, true = predict(
        model, make_loader(data.x_test, data.y_test, batch_size=256, shuffle=False), device
    )
    # Back to millivolts for delineation: undo the per-lead standardization.
    std = data.standardizer
    signals_mv = (
        data.x_test * std.std[None, :, None] + std.mean[None, :, None] if std else data.x_test
    )

    random_model = build_model(config["model"]).to(device).eval()  # untrained, random weights
    rng = new_generator(seed)
    segment_rows, lead_rows, sanity_rows = [], [], []
    examples: dict[str, np.ndarray] = {}
    ensure_dir(args.figures)

    for k, name in enumerate(SUPERCLASSES):
        detected = np.flatnonzero((true[:, k] == 1) & (prob[:, k] >= thresholds[k]))
        chosen = np.sort(
            rng.choice(detected, size=min(args.per_class, len(detected)), replace=False)
        )
        x = torch.from_numpy(data.x_test[chosen]).to(device)
        maps = integrated_gradients(model, x, k, steps=args.steps)
        random_maps = integrated_gradients(random_model, x, k, steps=args.steps)
        cams = grad_cam(model, x, k, model.blocks[-1])
        random_cams = grad_cam(random_model, x, k, random_model.blocks[-1])
        # Every map is also computed for the untrained model: its enrichment is
        # what the input's shape alone produces, the baseline to compare with.
        method_maps = {
            "integrated_gradients": maps,
            "integrated_gradients_random": random_maps,
            "grad_cam": cams,
            "grad_cam_random": random_cams,
        }

        method_shares: dict[str, list] = {method: [] for method in method_maps}
        delineated = []
        for i, record in enumerate(chosen):
            masks = beat_segments(signals_mv[record, LEAD_II], fs)
            if masks is None:
                continue
            delineated.append(i)
            for method, all_maps in method_maps.items():
                method_shares[method].append(segment_shares(all_maps[i], masks))
        print(f"{name}: {len(chosen)} detected records explained, {len(delineated)} delineated")

        for method, shares in method_shares.items():
            for segment, values in summarize_enrichment(shares).items():
                segment_rows.append(
                    {
                        "superclass": name,
                        "method": method,
                        "segment": segment,
                        "n_records": len(shares),
                    }
                    | {key: round(v, 4) for key, v in values.items()}
                )
        per_lead = np.nanmean([lead_shares(maps[i]) for i in range(len(chosen))], axis=0)
        lead_rows.append(
            {"superclass": name}
            | {lead: round(float(s), 4) for lead, s in zip(LEAD_NAMES, per_lead, strict=True)}
        )
        sanity_rows.append(
            {
                "superclass": name,
                "n_records": len(chosen),
                "integrated_gradients": round(
                    float(np.nanmedian(randomization_similarity(maps, random_maps))), 4
                ),
                "grad_cam": round(
                    float(np.nanmedian(randomization_similarity(cams, random_cams))), 4
                ),
                # Random-weight Grad-CAM is often all zero after its ReLU; its
                # enrichment is only meaningful when most maps are not.
                "grad_cam_random_nonzero": round(float((random_cams.max(axis=1) > 0).mean()), 4),
            }
        )

        # The most confident delineated record is this class's example.
        best = delineated[int(np.argmax(prob[chosen[delineated], k]))]
        record = chosen[best]
        masks = beat_segments(signals_mv[record, LEAD_II], fs)
        fig = plot_example(
            signals_mv[record], maps[best], masks, fs,
            f"{name}: test record {record}, p = {prob[record, k]:.2f}",
        )  # fmt: skip
        fig.savefig(args.figures / f"saliency_{name}.png", dpi=120)
        plt.close(fig)
        examples[f"{name}_signal"] = signals_mv[record].astype(np.float32)
        examples[f"{name}_ig"] = maps[best].astype(np.float32)
        examples[f"{name}_probability"] = np.float32(prob[record, k])
        for segment in SEGMENTS:
            examples[f"{name}_{segment}"] = masks[segment]

    tables = ensure_dir(args.tables)
    pd.DataFrame(segment_rows).to_csv(tables / "saliency_segments.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(tables / "saliency_leads.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(tables / "saliency_sanity.csv", index=False)
    ensure_dir(args.examples.parent)
    np.savez_compressed(args.examples, fs=fs, leads=np.array(LEAD_NAMES), **examples)

    summary = pd.DataFrame(segment_rows)
    for method in ("integrated_gradients", "grad_cam"):
        table = summary[summary["method"].str.startswith(method)]
        print(f"\n{method} enrichment (trained / random weights):")
        print(
            table.pivot(index="superclass", columns=["segment", "method"], values="enrichment")
            .round(2)
            .to_string()
        )
    print(pd.DataFrame(sanity_rows).to_string(index=False))
    print(f"Tables written to {tables}, figures to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
