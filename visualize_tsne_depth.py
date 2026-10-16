"""
t-SNE visualization of the DEPTH-BRANCH feature space, outdoor images only.

Directly relevant to this project's core contribution (distance-based reliability
weighting, see methodology Section 3.5): if the depth decoder's own internal features
naturally separate by ground-truth distance, that's independent evidence supporting
the premise that "near" and "far" outdoor pixels are genuinely different regimes for
the network -- not just an assumption baked into the loss function after the fact.

Produces two plots, saved to --out (default tsne_depth/):

  1. tsne_outdoor_depth_by_distance.png
     t-SNE of the pooled depth-branch feature (taken after cross-attention fusion +
     FiLM domain conditioning, before the DPT decoder -- see models/full_model.py's
     "depth_feat_pooled"), colored by each image's mean ground-truth outdoor depth.
     Continuous colormap (near=dark, far=bright) -- look for whether points form a
     gradient/cluster structure along the color axis, not just random scatter.

  2. tsne_outdoor_depth_by_reliability_bucket.png
     Same t-SNE coordinates, recolored into 3 discrete buckets matching the
     reliability-weighting regions from Section 3.5: "near, full trust" (<= near_distance),
     "transition" (near_distance to the floor-reached distance), and "far, floor weight"
     (beyond that). Useful for a paper figure precisely because it uses the same
     thresholds already justified in the methodology, rather than arbitrary bins.

Usage:
    python visualize_tsne_depth.py --ckpt checkpoints/best.pt
    python visualize_tsne_depth.py --ckpt checkpoints/swa_averaged.pt --max_samples 1000
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import data_cfg, model_cfg, train_cfg, loss_cfg
from data.dataset import build_dataloaders
from models.full_model import MultiTaskModel


@torch.no_grad()
def collect_outdoor_depth_features(model, loader, device, max_samples):
    model.eval()
    feats, mean_depths = [], []
    for batch in tqdm(loader, desc="extracting outdoor depth features", dynamic_ncols=True):
        if len(feats) >= max_samples:
            break
        domain_gt = batch["domain"]
        outdoor_mask = (domain_gt == 1)
        if outdoor_mask.sum() == 0:
            continue
        images = batch["image"].to(device, non_blocking=True)
        depth_gt = batch["depth"]
        depth_valid = batch["depth_valid"]

        outputs = model(images, domain_gt=None, teacher_force_domain=False, return_features=True)
        dfeat = outputs["depth_feat_pooled"].cpu().numpy()

        for i in range(images.shape[0]):
            if not outdoor_mask[i]:
                continue
            if len(feats) >= max_samples:
                break
            valid = depth_valid[i].numpy().astype(bool)
            if valid.sum() == 0:
                continue
            mean_d = float(depth_gt[i].numpy()[valid].mean())
            feats.append(dfeat[i])
            mean_depths.append(mean_d)

    return np.array(feats), np.array(mean_depths)


def reliability_bucket(mean_depths, near_distance, power, min_weight):
    """Same thresholds as depth_reliability_weight() in utils/losses.py, so the
    figure's bucket boundaries match exactly what's reported in the methodology."""
    floor_reached_at = near_distance / (min_weight ** (1.0 / power))
    buckets = np.zeros(len(mean_depths), dtype=int)
    buckets[mean_depths <= near_distance] = 0        # near, full trust
    buckets[(mean_depths > near_distance) & (mean_depths <= floor_reached_at)] = 1  # transition
    buckets[mean_depths > floor_reached_at] = 2       # far, floor weight
    return buckets, floor_reached_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--out", default="tsne_depth")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"

    model_cfg.pretrained = False  # loading real trained weights next, no need for ImageNet download
    model = MultiTaskModel(model_cfg, data_cfg.num_seg_classes, img_size=data_cfg.img_size).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")

    _, val_loader = build_dataloaders(data_cfg, train_cfg.batch_size, data_cfg.num_workers)
    loader = val_loader if args.split == "val" else _

    feats, mean_depths = collect_outdoor_depth_features(model, loader, device, args.max_samples)
    print(f"Collected {len(feats)} outdoor samples, depth range {mean_depths.min():.1f}-{mean_depths.max():.1f}m")

    if len(feats) < 10:
        raise SystemExit("Too few outdoor samples collected -- check --split and dataset paths.")

    print("Running t-SNE...")
    coords = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, len(feats) // 4)).fit_transform(feats)

    # ---- plot 1: continuous distance coloring ----
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mean_depths, cmap="viridis", s=28, alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean ground-truth depth (m)", fontsize=14)
    ax.set_title("Outdoor Depth-Branch Features (t-SNE)\ncolored by ground-truth distance", fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "tsne_outdoor_depth_by_distance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- plot 2: reliability-weighting bucket coloring ----
    buckets, floor_reached_at = reliability_bucket(
        mean_depths,
        getattr(loss_cfg, "depth_reliability_near", 15.0),
        getattr(loss_cfg, "depth_reliability_power", 2.0),
        getattr(loss_cfg, "depth_reliability_min_weight", 0.15),
    )
    labels = [f"Near (\u2264{loss_cfg.depth_reliability_near:.0f}m, full trust)",
              f"Transition ({loss_cfg.depth_reliability_near:.0f}-{floor_reached_at:.0f}m)",
              f"Far (>{floor_reached_at:.0f}m, floor weight)"]
    colors = ["#2166AC", "#F4A582", "#B2182B"]

    fig, ax = plt.subplots(figsize=(9, 8))
    for b in range(3):
        mask = buckets == b
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[b], s=28, alpha=0.85, label=f"{labels[b]} (n={mask.sum()})")
    ax.legend(fontsize=11, loc="best")
    ax.set_title("Outdoor Depth-Branch Features (t-SNE)\ncolored by reliability-weighting bucket", fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "tsne_outdoor_depth_by_reliability_bucket.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved both plots to {args.out}/")


if __name__ == "__main__":
    main()
