"""
Reads runs/metrics.csv (written incrementally by train.py via utils/csv_logger.py)
and saves SEPARATE per-domain, train-vs-val curve PNGs:

    curves/indoor_segmentation_miou.png
    curves/indoor_depth_rmse.png
    curves/indoor_depth_rmse_log.png
    curves/indoor_depth_delta.png        (delta1/delta2/delta3 together)
    curves/indoor_depth_absrel.png
    curves/indoor_depth_sqrel.png
    curves/outdoor_*.png                 (same six, for outdoor)

Each PNG overlays train (solid) and val (dashed) so you can see the
train/val gap per domain per metric directly.

Usage:
    python plot_curves.py                       # uses runs/metrics.csv, saves to curves/
    python plot_curves.py --csv runs/metrics.csv --out curves
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLOTS = [
    # (filename_suffix, title, [(column, label), ...], ylabel)
    ("segmentation_miou", "Segmentation mIoU", [("mIoU", "mIoU")], "mIoU"),
    ("depth_rmse", "Depth RMSE", [("RMSE", "RMSE (m)")], "RMSE (meters)"),
    ("depth_rmse_log", "Depth RMSE (log)", [("RMSE_log", "RMSE_log")], "RMSE (log space)"),
    ("depth_delta", "Depth accuracy (delta thresholds)",
     [("delta1", "delta1 (<1.25)"), ("delta2", "delta2 (<1.25^2)"), ("delta3", "delta3 (<1.25^3)")],
     "fraction of pixels"),
    ("depth_absrel", "Depth AbsRel", [("AbsRel", "AbsRel")], "AbsRel"),
    ("depth_sqrel", "Depth SqRel", [("SqRel", "SqRel")], "SqRel"),
]
# outdoor_capped is a VAL-ONLY, DEPTH-ONLY reporting metric (see LossConfig.depth_eval_cap_outdoor /
# train.py's outdoor_capped_metric) -- there's no segmentation data or train-split data for it, so it
# gets every plot above EXCEPT segmentation_miou.
DEPTH_ONLY_PLOTS = [p for p in PLOTS if p[0] != "segmentation_miou"]


def plot_domain_metric(df, domain, suffix, title, series, ylabel, out_dir):
    d = df[df["domain"] == domain]
    train_d = d[d["split"] == "train"].sort_values("epoch")
    val_d = d[d["split"] == "val"].sort_values("epoch")

    if train_d.empty and val_d.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (col, label) in enumerate(series):
        c = colors[i % len(colors)]
        if not train_d.empty and col in train_d:
            ax.plot(train_d["epoch"], train_d[col], linestyle="-", color=c, label=f"train {label}")
        if not val_d.empty and col in val_d:
            ax.plot(val_d["epoch"], val_d[col], linestyle="--", color=c, label=f"val {label}")

    domain_label = "Outdoor (gt<50m)" if domain == "outdoor_capped" else domain.capitalize()
    ax.set_title(f"{domain_label} -- {title} (train vs val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{domain}_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_curve_plots(csv_path="runs/metrics.csv", out_dir="curves"):
    """Callable entry point (used by train.py at the end of training, and by the CLI below)."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run train.py first (it writes this CSV every epoch), "
            f"or pass the right csv_path."
        )

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    saved = []
    for domain in ("indoor", "outdoor"):
        for suffix, title, series, ylabel in PLOTS:
            path = plot_domain_metric(df, domain, suffix, title, series, ylabel, out_dir)
            if path:
                saved.append(path)
    # outdoor_capped: depth-only, val-only (no train-split, no segmentation data for this domain --
    # plot_domain_metric already handles an empty train_d gracefully, drawing only the val line)
    for suffix, title, series, ylabel in DEPTH_ONLY_PLOTS:
        path = plot_domain_metric(df, "outdoor_capped", suffix, title, series, ylabel, out_dir)
        if path:
            saved.append(path)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/metrics.csv")
    ap.add_argument("--out", default="curves")
    args = ap.parse_args()

    saved = generate_curve_plots(args.csv, args.out)
    print(f"Saved {len(saved)} curve PNGs to {args.out}/:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
