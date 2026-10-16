"""
scan_dataset_outliers.py
==========================
Scans your ALREADY-PROCESSED dataset (the output of prepare_dataset.py) for
samples that are likely corrupted, degenerate, or otherwise anomalous -- the
"missing/bad data" side of robustness, distinct from the robust LOSS functions
in utils/losses.py (which handle noisy-but-present labels/depth during
training). This script doesn't touch training at all; it's a read-only report
so you know whether such samples exist in your data and roughly how many.

What it flags, per sample:
  - seg map almost entirely ignore_index (>90% by default) -- likely a
    corrupted/empty label, contributing near-zero real learning signal while
    still costing a full forward/backward pass
  - depth map almost entirely invalid (0 or out of [min_depth,max_depth])
  - depth map with suspiciously little variation (near-constant depth across
    the whole image -- often indicates a sensor/processing glitch, real scenes
    are essentially never uniform depth)
  - image file that fails to load at all (should already be impossible after
    prepare_dataset.py's None-checks, but verified here independently)

This does NOT modify or delete anything -- it only reports. Deciding whether
to exclude flagged samples (and how) is a judgment call once you've seen
what's actually there; for most datasets a handful of edge cases is normal
and not worth acting on, but a large or systematic cluster would be worth
investigating (e.g. a whole scene/city misprocessed).

Usage:
    python scan_dataset_outliers.py --root /home/hashir/Thesis/Dataset/processed --split train
    python scan_dataset_outliers.py --root /home/hashir/Thesis/Dataset/processed --split val
"""
import argparse
import json
import os

import cv2
import numpy as np


def scan_split(root, split, ignore_index=255, min_depth=0.001, max_depth=80.0,
                seg_ignore_frac_threshold=0.90, depth_invalid_frac_threshold=0.90,
                depth_std_threshold=0.05):
    img_dir = os.path.join(root, split, "images")
    seg_dir = os.path.join(root, split, "seg")
    depth_dir = os.path.join(root, split, "depth")
    domain_file = os.path.join(root, split, "domain.txt")

    domain_map = {}
    with open(domain_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n, d = line.rsplit(" ", 1)
            domain_map[n] = int(d)

    names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))

    flags = {"unreadable": [], "degenerate_seg": [], "degenerate_depth_invalid": [],
             "degenerate_depth_flat": []}
    domain_counts = {0: 0, 1: 0}

    for name in names:
        domain = domain_map.get(name, -1)
        if domain in (0, 1):
            domain_counts[domain] += 1

        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(img_dir, name + ext)
            if os.path.isfile(p):
                img_path = p
                break
        seg_path = os.path.join(seg_dir, name + ".png")
        depth_path = os.path.join(depth_dir, name + ".png")

        img = cv2.imread(img_path) if img_path else None
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if img is None or seg is None or depth_raw is None:
            flags["unreadable"].append(name)
            continue

        ignore_frac = np.mean(seg == ignore_index)
        if ignore_frac > seg_ignore_frac_threshold:
            flags["degenerate_seg"].append((name, round(float(ignore_frac), 3)))

        depth_m = depth_raw.astype(np.float32) / 1000.0
        valid = (depth_m > min_depth) & (depth_m < max_depth)
        invalid_frac = 1.0 - np.mean(valid)
        if invalid_frac > depth_invalid_frac_threshold:
            flags["degenerate_depth_invalid"].append((name, round(float(invalid_frac), 3)))
        elif valid.sum() > 0:
            depth_std = float(np.std(depth_m[valid]))
            if depth_std < depth_std_threshold:
                flags["degenerate_depth_flat"].append((name, round(depth_std, 4)))

    return names, domain_counts, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seg_ignore_frac_threshold", type=float, default=0.90)
    ap.add_argument("--depth_invalid_frac_threshold", type=float, default=0.90)
    ap.add_argument("--depth_std_threshold", type=float, default=0.05)
    ap.add_argument("--save_json", default="",
                     help="optional path to save the full flagged-name lists as JSON")
    args = ap.parse_args()

    names, domain_counts, flags = scan_split(
        args.root, args.split,
        seg_ignore_frac_threshold=args.seg_ignore_frac_threshold,
        depth_invalid_frac_threshold=args.depth_invalid_frac_threshold,
        depth_std_threshold=args.depth_std_threshold,
    )

    n_total = len(names)
    print(f"\n===== {args.split} split: {n_total} total samples "
          f"(indoor={domain_counts[0]}, outdoor={domain_counts[1]}) =====\n")

    for key, label in [
        ("unreadable", "Unreadable (file exists but failed to decode)"),
        ("degenerate_seg", f"Segmentation map >{args.seg_ignore_frac_threshold*100:.0f}% ignore_index"),
        ("degenerate_depth_invalid", f"Depth map >{args.depth_invalid_frac_threshold*100:.0f}% invalid/out-of-range"),
        ("degenerate_depth_flat", f"Depth map suspiciously flat (std < {args.depth_std_threshold}m)"),
    ]:
        n_flagged = len(flags[key])
        pct = 100 * n_flagged / max(n_total, 1)
        print(f"{label}: {n_flagged} samples ({pct:.2f}%)")
        if n_flagged > 0:
            sample_preview = flags[key][:5]
            print(f"    example(s): {sample_preview}")

    total_flagged = len(set(
        [n for n in flags["unreadable"]] +
        [n for n, _ in flags["degenerate_seg"]] +
        [n for n, _ in flags["degenerate_depth_invalid"]] +
        [n for n, _ in flags["degenerate_depth_flat"]]
    ))
    print(f"\nTotal unique flagged samples: {total_flagged} / {n_total} "
          f"({100*total_flagged/max(n_total,1):.2f}%)")
    print("\nThis is a REPORT ONLY -- nothing was modified. A small fraction (a few percent)")
    print("of edge cases is normal for real-world datasets and not necessarily worth acting")
    print("on. Worth a closer look if: a large fraction is flagged, or flagged samples")
    print("cluster suspiciously (e.g. all from one scene/city -- check the names above).")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(flags, f, indent=2)
        print(f"\nFull flagged lists saved to {args.save_json}")


if __name__ == "__main__":
    main()
