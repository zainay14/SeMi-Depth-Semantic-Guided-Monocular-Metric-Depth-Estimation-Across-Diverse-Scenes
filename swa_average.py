"""
Stochastic Weight Averaging (SWA) -- Izmailov et al. 2018, "Averaging Weights
Leads to Wider Optima and Better Generalization".

Standard, well-established technique: plain arithmetic averaging of model weights
from several late-training epochs, after the loss has roughly plateaued. Often
gives a genuine, free 1-3% improvement in generalization metrics (i.e. exactly
the val-side gap we've been fighting) -- because the average of several nearby
"good" weight settings tends to sit in a wider, flatter, better-generalizing
region of the loss landscape than any single one of them.

This is NOT a trick to inflate a metric -- it produces one real, ordinary set of
model weights, exactly as loadable/usable/citable as any other checkpoint. It's
standard enough that PyTorch ships torch.optim.swa_utils built specifically for
this. Report it as "SWA over epochs X-Y" in your methods section, same as anyone
citing this technique would.

Usage (after training has run past config.py's swa_start_epoch for a while,
ideally at least 5-10 snapshots):
    python swa_average.py
    python swa_average.py --snapshot_dir checkpoints/swa_snapshots --out checkpoints/swa_averaged.pt

Then evaluate it exactly like any other checkpoint:
    python evaluate.py --ckpt checkpoints/swa_averaged.pt --tta
"""
import argparse
import glob
import os

import torch

from config import data_cfg, model_cfg, train_cfg
from models.full_model import MultiTaskModel


def average_checkpoints(snapshot_paths):
    """Plain arithmetic mean of every tensor in the state dicts. BatchNorm running
    stats get averaged too, which is the simple/standard approach -- if you want
    the more careful BN-recalibration variant, run one extra forward pass over the
    training set in train mode after loading the averaged weights (not done here
    to keep this a pure post-hoc, no-extra-training step)."""
    avg_state = None
    n = len(snapshot_paths)
    for path in snapshot_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_state[k] += v.float()
    for k in avg_state:
        avg_state[k] /= n
    return avg_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot_dir", default=os.path.join(train_cfg.ckpt_dir, "swa_snapshots"))
    ap.add_argument("--out", default=os.path.join(train_cfg.ckpt_dir, "swa_averaged.pt"))
    ap.add_argument("--last_n", type=int, default=0,
                     help="only average the LAST N snapshots (by epoch number). 0 = use all found.")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.snapshot_dir, "epoch_*.pt")),
                    key=lambda p: int(os.path.basename(p).replace("epoch_", "").replace(".pt", "")))
    if not paths:
        raise SystemExit(
            f"No snapshots found in {args.snapshot_dir}. These are only saved from "
            f"train_cfg.swa_start_epoch (currently {train_cfg.swa_start_epoch}) onward -- "
            f"if you haven't trained that far yet, there's nothing to average yet."
        )
    if args.last_n > 0:
        paths = paths[-args.last_n:]

    epochs = [int(os.path.basename(p).replace("epoch_", "").replace(".pt", "")) for p in paths]
    print(f"Averaging {len(paths)} snapshots, epochs {epochs}")

    avg_state = average_checkpoints(paths)

    # sanity check: build a real model and confirm the averaged weights actually load.
    # pretrained=False here deliberately -- we're about to overwrite every weight with
    # the averaged state dict anyway, so downloading real ImageNet pretrained weights
    # first would be pure wasted time (and a needless network dependency/failure point).
    model_cfg.pretrained = False
    model = MultiTaskModel(model_cfg, data_cfg.num_seg_classes, img_size=data_cfg.img_size)
    model.load_state_dict(avg_state)
    print("Averaged weights loaded into model successfully -- shapes all match.")

    torch.save({"model": avg_state, "epoch": f"swa_avg_{epochs[0]}-{epochs[-1]}"}, args.out)
    print(f"Saved SWA-averaged checkpoint to {args.out}")
    print(f"Evaluate it with: python evaluate.py --ckpt {args.out} --tta")


if __name__ == "__main__":
    main()
