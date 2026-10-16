"""
MultiTaskDataset
=================
Expects the following folder layout (see README.md / prepare_dataset.py):

    dataset_root/
        train/
            images/    <name>.jpg|png            RGB image
            seg/       <name>.png                 uint8 segmentation label map (class ids, ignore=255)
            depth/     <name>.png                 uint16 depth map in millimeters (0 = invalid)
            domain.txt                             "<name> <0|1>" per line   (0=indoor, 1=outdoor)
        val/
            ... same structure ...

This layout is deliberately dataset-agnostic: it is what you get after running
data/prepare_dataset.py on NYU-Depth-V2 (indoor) + Cityscapes-with-disparity or
Virtual KITTI 2 (outdoor). Any dataset can be adapted to it.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _validate_num_seg_classes(root, num_seg_classes):
    """
    Catches the exact bug this pipeline shipped with: config.py's num_seg_classes
    silently drifting out of sync with data/prepare_dataset.py's UNIFIED_CLASSES
    (which writes classes.json). If classes.json is present, fail loudly instead
    of quietly training a seg head with the wrong number of output channels.
    """
    classes_path = os.path.join(root, "classes.json")
    if not os.path.isfile(classes_path):
        return
    import json
    with open(classes_path) as f:
        classes = json.load(f)["classes"]
    if len(classes) != num_seg_classes:
        raise ValueError(
            f"config.py DataConfig.num_seg_classes={num_seg_classes} but {classes_path} "
            f"lists {len(classes)} classes: {classes}. These MUST match -- the seg head's "
            f"output channels, the loss, and mIoU are all sized from num_seg_classes. "
            f"Fix config.py's num_seg_classes to {len(classes)}."
        )


class MultiTaskDataset(Dataset):
    def __init__(self, root, split, img_size, num_seg_classes, ignore_index,
                 min_depth, max_depth, depth_scale, mean, std, transforms,
                 minority_transforms=None, minority_domain=None,
                 subsample_indoor_stride=1):
        """
        minority_transforms / minority_domain: if set, samples whose domain label
        equals `minority_domain` are augmented with `minority_transforms` instead of
        `transforms` (see utils/transforms.build_train_transforms_strong). Only
        meaningful for the train split; leave both None for standard behavior.
        """
        self.split_dir = os.path.join(root, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.seg_dir = os.path.join(self.split_dir, "seg")
        self.depth_dir = os.path.join(self.split_dir, "depth")
        self.ignore_index = ignore_index
        self.num_seg_classes = num_seg_classes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.transforms = transforms
        self.minority_transforms = minority_transforms
        self.minority_domain = minority_domain

        _validate_num_seg_classes(root, num_seg_classes)
        assert os.path.isdir(self.img_dir), f"Missing images dir: {self.img_dir}"
        self.names = sorted(
            os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # domain labels: name -> 0/1
        self.domain_map = {}
        domain_file = os.path.join(self.split_dir, "domain.txt")
        if os.path.isfile(domain_file):
            with open(domain_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n, d = line.rsplit(" ", 1)
                    self.domain_map[n] = int(d)
        else:
            raise FileNotFoundError(
                f"{domain_file} not found. Every split needs a domain.txt mapping "
                f"'<image_name> <0|1>' (0=indoor, 1=outdoor)."
            )

        # Optional indoor subsampling for training efficiency (skips redundant consecutive video frames)
        if split == "train" and subsample_indoor_stride > 1:
            filtered_names = []
            indoor_count = 0
            for n in self.names:
                d = self.domain_map.get(n, 0)
                if d == 1:  # outdoor (Cityscapes) -> keep 100%
                    filtered_names.append(n)
                else:       # indoor (ScanNet video) -> subsample by stride
                    if indoor_count % subsample_indoor_stride == 0:
                        filtered_names.append(n)
                    indoor_count += 1
            print(f"[dataset] train split indoor subsampled by {subsample_indoor_stride}x: "
                  f"{len(self.names)} -> {len(filtered_names)} images (outdoor 100% preserved)")
            self.names = filtered_names

        # per-sample domain label list, same order as self.names -- used by
        # build_dataloaders() to construct a class-balanced sampler.
        self.domains = [self.domain_map.get(n, 0) for n in self.names]

    def __len__(self):
        return len(self.names)

    def _find(self, folder, name):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(folder, name + ext)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No file for {name} in {folder}")

    def __getitem__(self, idx):
        name = self.names[idx]

        img = cv2.imread(self._find(self.img_dir, name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seg = cv2.imread(self._find(self.seg_dir, name), cv2.IMREAD_UNCHANGED).astype(np.int64)
        if seg.ndim == 3:
            seg = seg[..., 0]

        depth_raw = cv2.imread(self._find(self.depth_dir, name), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_raw / self.depth_scale  # -> meters
        valid = (depth_m > self.min_depth) & (depth_m < self.max_depth)
        depth_m = np.clip(depth_m, self.min_depth, self.max_depth)

        domain = self.domain_map.get(name, 0)
        active_transforms = self.transforms
        if self.minority_transforms is not None and domain == self.minority_domain:
            active_transforms = self.minority_transforms

        # pack validity into the depth channel's sign trick avoided; keep a separate mask
        augmented = active_transforms(image=img, mask=seg, depth=depth_m)
        img_t = augmented["image"]                      # (3,H,W) float, normalized
        seg_t = augmented["mask"].long()                 # (H,W)
        depth_t = augmented["depth"].float()              # (H,W) meters

        # recompute validity mask after geometric transform (padding introduces 0s -> invalid)
        valid_t = (depth_t > self.min_depth) & (depth_t < self.max_depth)
        seg_t[seg_t >= self.num_seg_classes] = self.ignore_index

        return {
            "image": img_t,
            "seg": seg_t,
            "depth": depth_t,
            "depth_valid": valid_t,
            "domain": torch.tensor(domain, dtype=torch.long),
            "name": name,
        }


def build_dataloaders(data_cfg, batch_size, num_workers, balanced_domain_sampling=True):
    from utils.transforms import build_train_transforms, build_train_transforms_strong, build_val_transforms
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from collections import Counter

    minority_transforms, minority_domain = None, None
    if getattr(data_cfg, "strong_aug_minority_domain", False):
        # Determine the minority domain from domain.txt directly (cheap, no image I/O)
        # so we know which domain to route through the stronger augmentation pipeline.
        probe_map = {}
        domain_file = os.path.join(data_cfg.root, data_cfg.train_split, "domain.txt")
        if os.path.isfile(domain_file):
            with open(domain_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n, d = line.rsplit(" ", 1)
                    probe_map[n] = int(d)
        if probe_map:
            counts = Counter(probe_map.values())
            minority_domain = min(counts, key=counts.get)
            minority_transforms = build_train_transforms_strong(
                data_cfg.img_size, data_cfg.mean, data_cfg.std, ignore_index=data_cfg.ignore_index)
            print(f"[augmentation] minority domain = {minority_domain} "
                  f"({counts[minority_domain]} imgs, vs {dict(counts)} total) "
                  f"-> using stronger augmentation for it")

    train_ds = MultiTaskDataset(
        root=data_cfg.root, split=data_cfg.train_split, img_size=data_cfg.img_size,
        num_seg_classes=data_cfg.num_seg_classes, ignore_index=data_cfg.ignore_index,
        min_depth=data_cfg.min_depth, max_depth=data_cfg.max_depth, depth_scale=data_cfg.depth_scale,
        mean=data_cfg.mean, std=data_cfg.std,
        transforms=build_train_transforms(data_cfg.img_size, data_cfg.mean, data_cfg.std,
                                           ignore_index=data_cfg.ignore_index),
        minority_transforms=minority_transforms, minority_domain=minority_domain,
        subsample_indoor_stride=getattr(data_cfg, "subsample_indoor_stride", 1),
    )
    val_ds = MultiTaskDataset(
        root=data_cfg.root, split=data_cfg.val_split, img_size=data_cfg.img_size,
        num_seg_classes=data_cfg.num_seg_classes, ignore_index=data_cfg.ignore_index,
        min_depth=data_cfg.min_depth, max_depth=data_cfg.max_depth, depth_scale=data_cfg.depth_scale,
        mean=data_cfg.mean, std=data_cfg.std,
        transforms=build_val_transforms(data_cfg.img_size, data_cfg.mean, data_cfg.std),
    )

    if balanced_domain_sampling:
        # Inverse-frequency weighting so each domain (indoor/outdoor) is seen with
        # closer-to-equal probability per epoch, regardless of how many raw images
        # each source dataset (NYU vs Cityscapes) contributed. This does NOT
        # delete/subsample data -- it's sampling-with-replacement, so minority-domain
        # images are simply revisited more often per epoch.
        #
        # CAPPED at max_domain_oversample_ratio (not full-balanced to the majority's
        # count): full balancing revisits each outdoor image ~13.5x/epoch (to match
        # indoor's ~40140 vs outdoor's ~2975), which is a lot of repetition of the
        # exact same images and a real contributor to the outdoor train/val RMSE gap
        # we diagnosed (train RMSE improving steadily while val RMSE plateaued --
        # classic overfitting-via-repetition signature). Capping the ratio trades a
        # bit of domain balance for less memorization pressure.
        counts = Counter(train_ds.domains)
        print(f"[sampler] train domain counts: "
              f"{ {d: c for d, c in sorted(counts.items())} }")
        majority_domain = max(counts, key=counts.get)
        majority_count = counts[majority_domain]
        max_ratio = getattr(data_cfg, "max_domain_oversample_ratio", None)
        class_weight = {}
        for d, c in counts.items():
            full_balance_weight = 1.0 / c
            if max_ratio is not None:
                capped_weight = min(full_balance_weight, max_ratio / majority_count)
                class_weight[d] = capped_weight
            else:
                class_weight[d] = full_balance_weight
        effective_ratios = {d: class_weight[d] / class_weight[majority_domain] for d in counts}
        print(f"[sampler] effective per-image oversampling ratio relative to majority domain "
              f"({majority_domain}): {effective_ratios} "
              f"(capped at {max_ratio}x)" if max_ratio is not None else
              f"(full balance, uncapped)")
        sample_weights = [class_weight[d] for d in train_ds.domains]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader
