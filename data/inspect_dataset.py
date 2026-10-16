"""
inspect_dataset.py
===================
Run this FIRST, before prepare_dataset.py, to verify two things we can't see
from folder screenshots alone:

  1. ScanNet 'label' folder convention -- is it raw NYU40 ids (1-40, 0=void),
     the 20-class "scannet20"/"eigen13"-style remap, or something else?
     We need to know the id range to build a correct label -> unified mapping.

  2. Whether Cityscapes' camera.json calibration files are present. Without
     them we fall back to constant/approximate calibration for disparity ->
     depth conversion, which is less accurate.

Usage:
    python data/inspect_dataset.py \
        --cityscapes_root "C:/Users/USER/Desktop/New_Research/Thesis/Dataset/outdoor/gtFine_trainvaltest" \
        --cityscapes_disp_root "C:/.../disparity_trainvaltest" \
        --cityscapes_img_root "C:/.../leftImg8bit_trainvaltest" \
        --scannet_root "C:/Users/USER/Desktop/New_Research/Thesis/Dataset/indoor/archive (1)/scannet_2d"
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def inspect_scannet(root, n_scenes=3, n_frames=3):
    print("\n===== ScanNet =====")
    scenes = sorted(os.listdir(root))[:n_scenes]
    for scene in scenes:
        label_dir = os.path.join(root, scene, "label")
        depth_dir = os.path.join(root, scene, "depth")
        color_dir = os.path.join(root, scene, "color")
        if not os.path.isdir(label_dir):
            print(f"  [{scene}] no 'label' folder found -- check path")
            continue
        label_files = sorted(glob.glob(os.path.join(label_dir, "*")))[:n_frames]
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*")))[:n_frames]
        color_files = sorted(glob.glob(os.path.join(color_dir, "*")))[:n_frames]

        for lf in label_files:
            lbl = cv2.imread(lf, cv2.IMREAD_UNCHANGED)
            print(f"  [{scene}] {os.path.basename(lf)}: dtype={lbl.dtype}, "
                  f"shape={lbl.shape}, unique_vals(first 20)={np.unique(lbl)[:20]}, "
                  f"max={lbl.max()}")
        for df in depth_files:
            d = cv2.imread(df, cv2.IMREAD_UNCHANGED)
            print(f"  [{scene}] {os.path.basename(df)}: dtype={d.dtype}, "
                  f"shape={d.shape}, min={d.min()}, max={d.max()}, "
                  f"nonzero_frac={np.mean(d > 0):.3f}")
        for cf in color_files:
            c = cv2.imread(cf)
            print(f"  [{scene}] {os.path.basename(cf)}: shape={c.shape}")

    print("\n  -> If label max <= 40 and looks like small integers (0-40): likely raw NYU40 ids.")
    print("  -> If label max <= 20: likely the 'scannet20'/'eigen' remapped set (different mapping needed).")
    print("  -> If depth dtype is uint16 with values in the thousands: it's millimeters (matches depth_scale=1000).")


def inspect_cityscapes(gtfine_root, disp_root, img_root, n_samples=3):
    print("\n===== Cityscapes =====")
    train_img_dir = os.path.join(img_root, "leftImg8bit", "train")
    if not os.path.isdir(train_img_dir):
        print(f"  leftImg8bit/train not found under {img_root}")
        return
    cities = sorted(os.listdir(train_img_dir))[:1]
    count = 0
    for city in cities:
        city_dir = os.path.join(train_img_dir, city)
        for fname in sorted(os.listdir(city_dir)):
            if count >= n_samples:
                break
            if not fname.endswith("_leftImg8bit.png"):
                continue
            stem = fname.replace("_leftImg8bit.png", "")
            img_path = os.path.join(city_dir, fname)
            seg_path = os.path.join(gtfine_root, "gtFine", "train", city, f"{stem}_gtFine_labelIds.png")
            disp_path = os.path.join(disp_root, "disparity", "train", city, f"{stem}_disparity.png")
            cam_path = os.path.join(img_root, "camera", "train", city, f"{stem}_camera.json") \
                if os.path.isdir(os.path.join(img_root, "camera")) else \
                os.path.join(gtfine_root, "camera", "train", city, f"{stem}_camera.json")

            img = cv2.imread(img_path)
            print(f"  image: {fname}, shape={img.shape if img is not None else None}")

            if os.path.isfile(seg_path):
                seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                print(f"  seg (labelIds): dtype={seg.dtype}, unique={np.unique(seg)}")
            else:
                print(f"  MISSING seg file: {seg_path}")

            if os.path.isfile(disp_path):
                disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
                print(f"  disparity: dtype={disp.dtype}, min={disp.min()}, max={disp.max()}, "
                      f"nonzero_frac={np.mean(disp > 0):.3f}")
            else:
                print(f"  MISSING disparity file: {disp_path}")

            if os.path.isfile(cam_path):
                with open(cam_path) as f:
                    cam = json.load(f)
                print(f"  camera.json FOUND: baseline={cam['extrinsic']['baseline']}, fx={cam['intrinsic']['fx']}")
            else:
                print(f"  camera.json NOT FOUND at {cam_path} "
                      f"-- you likely need to also download camera_trainvaltest.zip")
            count += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes_root", help="path to the extracted gtFine_trainvaltest folder (contains gtFine/)")
    ap.add_argument("--cityscapes_disp_root", help="path to the extracted disparity_trainvaltest folder (contains disparity/)")
    ap.add_argument("--cityscapes_img_root", help="path to the extracted leftImg8bit_trainvaltest folder (contains leftImg8bit/, and camera/ if downloaded)")
    ap.add_argument("--scannet_root", help="path to scannet_2d folder (contains scene0000_00/, ...)")
    args = ap.parse_args()

    if args.scannet_root:
        inspect_scannet(args.scannet_root)
    if args.cityscapes_root and args.cityscapes_disp_root and args.cityscapes_img_root:
        inspect_cityscapes(args.cityscapes_root, args.cityscapes_disp_root, args.cityscapes_img_root)
