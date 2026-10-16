"""
t-SNE visualization of learned features, run AFTER training on a saved checkpoint.

Produces two kinds of plots, saved to --out (default tsne/):

  1. tsne_domain_separation.png
     One t-SNE of the pooled bottleneck feature (domain head's input embedding,
     "domain_feat") for a mix of indoor + outdoor samples, colored by domain.
     Shows whether the shared Swin encoder actually separates indoor vs outdoor
     in feature space (this is what the domain head classifies from).

  2. tsne_indoor_classes.png / tsne_outdoor_classes.png
     TWO SEPARATE t-SNE plots (one indoor-only, one outdoor-only), each on the
     pooled segmentation feature ("seg_feat_pooled"), colored by each image's
     dominant (majority-pixel) segmentation class. Shows semantic clustering
     WITHIN each domain separately, i.e. "t-SNE of both datasets separately".

Usage:
    python visualize_tsne.py --ckpt checkpoints/best.pt
    python visualize_tsne.py --ckpt checkpoints/best.pt --split val --max_per_domain 800
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import data_cfg, model_cfg, train_cfg
from data.dataset import build_dataloaders
from models.full_model import MultiTaskModel

# kept in sync with data/prepare_dataset.py's UNIFIED_CLASSES ordering
try:
    with open(os.path.join(data_cfg.root, data_cfg.val_split, "..", "classes.json")) as f:
        CLASS_NAMES = json.load(f)["classes"]
except Exception:
    CLASS_NAMES = None  # falls back to plain integer labels if classes.json isn't found


@torch.no_grad()
def collect_features(model, loader, device, max_per_domain):
    model.eval()
    domain_feats, domain_labels = [], []
    seg_feats, seg_dom, seg_dom_class = [], [], []
    counts = {0: 0, 1: 0}

    for batch in tqdm(loader, desc="extracting features", dynamic_ncols=True):
        if counts[0] >= max_per_domain and counts[1] >= max_per_domain:
            break
        images = batch["image"].to(device, non_blocking=True)
        seg_gt = batch["seg"]
        domain_gt = batch["domain"]

        outputs = model(images, domain_gt=None, teacher_force_domain=False, return_features=True)
        dfeat = outputs["domain_feat"].cpu().numpy()
        sfeat = outputs["seg_feat_pooled"].cpu().numpy()

        for i in range(images.shape[0]):
            d = int(domain_gt[i].item())
            if counts[d] >= max_per_domain:
                continue
            counts[d] += 1
            domain_feats.append(dfeat[i])
            domain_labels.append(d)
            seg_feats.append(sfeat[i])
            seg_dom.append(d)

            # dominant (majority, ignoring 255) segmentation class in this image
            s = seg_gt[i].numpy()
            valid = s[s != data_cfg.ignore_index]
            if valid.size == 0:
                seg_dom_class.append(-1)
            else:
                seg_dom_class.append(int(np.bincount(valid).argmax()))

    return (np.stack(domain_feats), np.array(domain_labels),
            np.stack(seg_feats), np.array(seg_dom), np.array(seg_dom_class))


def run_tsne(features, perplexity=30, seed=42):
    n = features.shape[0]
    perplexity = min(perplexity, max(5, n // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed)
    return tsne.fit_transform(features)


def plot_domain_separation(emb, labels, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    names = {0: "indoor", 1: "outdoor"}
    colors = {0: "#1f77b4", 1: "#d62728"}
    for d in (0, 1):
        mask = labels == d
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, c=colors[d], label=names[d])
    ax.set_title("t-SNE of shared bottleneck feature, colored by domain")
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_within_domain_classes(emb, class_ids, domain_name, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    uniq = sorted(set(int(c) for c in class_ids if c >= 0))
    cmap = plt.get_cmap("tab20", max(len(uniq), 1))
    for i, c in enumerate(uniq):
        mask = class_ids == c
        label = CLASS_NAMES[c] if CLASS_NAMES and c < len(CLASS_NAMES) else str(c)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6, color=cmap(i), label=label)
    ax.set_title(f"t-SNE of {domain_name} segmentation features, colored by dominant class")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_tsne_visualization(model=None, ckpt_path=None, split="val", max_per_domain=1000,
                            out_dir="tsne", device=None):
    """
    Callable entry point. Either pass an already-loaded `model` (e.g. straight from
    train.py right after training finishes -- no need to reload weights from disk),
    or pass `ckpt_path` to load one fresh (used by the CLI below).
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    if model is None:
        model = MultiTaskModel(model_cfg, data_cfg.num_seg_classes, img_size=data_cfg.img_size).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    train_loader, val_loader = build_dataloaders(
        data_cfg, train_cfg.batch_size, data_cfg.num_workers, balanced_domain_sampling=False)
    loader = train_loader if split == "train" else val_loader

    domain_feats, domain_labels, seg_feats, seg_dom, seg_dom_class = collect_features(
        model, loader, device, max_per_domain)

    print(f"[tsne] collected {len(domain_labels)} samples "
          f"(indoor={int((domain_labels == 0).sum())}, outdoor={int((domain_labels == 1).sum())})")

    saved = []
    # 1. domain-separation t-SNE (both domains together, colored by domain)
    emb_domain = run_tsne(domain_feats)
    path = os.path.join(out_dir, "tsne_domain_separation.png")
    plot_domain_separation(emb_domain, domain_labels, path)
    saved.append(path)

    # 2. separate t-SNE per domain, colored by dominant segmentation class
    for d, name in ((0, "indoor"), (1, "outdoor")):
        mask = seg_dom == d
        if mask.sum() < 10:
            print(f"[tsne] skipping {name} class t-SNE: only {int(mask.sum())} samples")
            continue
        emb_d = run_tsne(seg_feats[mask])
        path = os.path.join(out_dir, f"tsne_{name}_classes.png")
        plot_within_domain_classes(emb_d, seg_dom_class[mask], name, path)
        saved.append(path)

    print(f"[tsne] saved {len(saved)} plots to {out_dir}/")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(train_cfg.ckpt_dir, "best.pt"))
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--max_per_domain", type=int, default=1000,
                     help="cap samples per domain (t-SNE is O(n^2)-ish; keep this reasonable)")
    ap.add_argument("--out", default="tsne")
    args = ap.parse_args()

    run_tsne_visualization(ckpt_path=args.ckpt, split=args.split,
                            max_per_domain=args.max_per_domain, out_dir=args.out)


if __name__ == "__main__":
    main()
