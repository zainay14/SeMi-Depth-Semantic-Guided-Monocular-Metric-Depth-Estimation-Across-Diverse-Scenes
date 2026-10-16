"""
prepare_dataset.py  (Cityscapes + ScanNet version)
====================================================
Converts:
    Cityscapes (outdoor, domain=1) -- gtFine_trainvaltest + leftImg8bit_trainvaltest
                                        + disparity_trainvaltest [+ camera_trainvaltest]
    ScanNet 2D (indoor, domain=0)  -- scannet_2d/scene####_##/{color,depth,label,pose}

into the unified layout MultiTaskDataset expects:
    dataset_root/{train,val}/{images,seg,depth}/<name>.{jpg,png}
    dataset_root/{train,val}/domain.txt

Run data/inspect_dataset.py FIRST to confirm the ScanNet label id convention
and whether Cityscapes camera.json calibration files are present -- both
assumptions below are configurable via CLI flags once you know.

Usage:
    python data/prepare_dataset.py \
        --cityscapes_gtfine   "D:/.../outdoor/gtFine_trainvaltest" \
        --cityscapes_img      "D:/.../outdoor/leftImg8bit_trainvaltest" \
        --cityscapes_disp     "D:/.../outdoor/disparity_trainvaltest" \
        --cityscapes_camera   "D:/.../outdoor/camera_trainvaltest"   (optional) \
        --scannet_root        "D:/.../indoor/archive (1)/scannet_2d" \
        --out ./dataset --target_short_side 480 --scannet_val_ratio 0.1
"""
import argparse
import json
import os
import random

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Unified label space shared by both domains. Edit freely -- indoor-only
# classes (floor_ground, ceiling, furniture, window, door) will simply never
# appear in outdoor samples and vice versa; that's fine, the loss is computed
# per-pixel regardless of which classes a given image happens to contain.
# ---------------------------------------------------------------------------
UNIFIED_CLASSES = [
    "floor_ground", "wall", "ceiling", "building", "road", "sidewalk",
    "vegetation", "sky", "furniture", "person", "vehicle", "window",
    "door", "object_other", "terrain",
]
IGNORE_INDEX = 255
U = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

# ---------------------------------------------------------------------------
# Cityscapes: gtFine_*_labelIds.png uses the RAW 34-class ids (0-33), NOT the
# 19-class trainIds. Full official id list: cityscapesscripts/helpers/labels.py
# ---------------------------------------------------------------------------
CITYSCAPES_RAW_ID_TO_UNIFIED = {
    7: U["road"], 8: U["sidewalk"], 9: U["road"],           # road, sidewalk, parking
    11: U["building"], 12: U["wall"], 13: U["object_other"],  # building, wall, fence
    14: U["object_other"], 15: U["building"], 16: U["building"],  # guard rail, bridge, tunnel
    17: U["object_other"], 19: U["object_other"], 20: U["object_other"],  # pole, traffic light, traffic sign
    21: U["vegetation"], 22: U["terrain"], 23: U["sky"],
    24: U["person"], 25: U["person"],                        # person, rider
    26: U["vehicle"], 27: U["vehicle"], 28: U["vehicle"],    # car, truck, bus
    29: U["vehicle"], 30: U["vehicle"], 31: U["vehicle"],    # caravan, trailer, train
    32: U["vehicle"], 33: U["vehicle"],                       # motorcycle, bicycle
    # everything else (0-6,10,18) -> ignore (ego vehicle, out-of-roi, static/dynamic,
    # ground, rail track, polegroup -- ambiguous / not useful classes)
}

# ---------------------------------------------------------------------------
# ScanNet: default assumption is the label PNG stores raw NYU40 ids (1-40,
# 0 = unlabeled/void). If data/inspect_dataset.py shows a different max value
# (e.g. <=20), pass --scannet_label_convention scannet20 and extend the
# SCANNET20_ID_TO_UNIFIED table below to match.
# ---------------------------------------------------------------------------
NYU40_ID_TO_UNIFIED = {
    1: U["wall"], 2: U["floor_ground"], 3: U["furniture"], 4: U["furniture"],
    5: U["furniture"], 6: U["furniture"], 7: U["furniture"], 8: U["door"],
    9: U["window"], 10: U["furniture"], 11: U["object_other"], 12: U["furniture"],
    13: U["window"], 14: U["furniture"], 15: U["furniture"], 16: U["window"],
    17: U["furniture"], 18: U["furniture"], 19: U["object_other"], 20: U["floor_ground"],
    21: U["object_other"], 22: U["ceiling"], 23: U["object_other"], 24: U["furniture"],
    25: U["object_other"], 26: U["object_other"], 27: U["object_other"], 28: U["window"],
    29: U["object_other"], 30: U["object_other"], 31: U["person"], 32: U["furniture"],
    33: U["furniture"], 34: U["furniture"], 35: U["object_other"], 36: U["furniture"],
    37: U["object_other"], 38: U["object_other"], 39: U["furniture"], 40: U["object_other"],
    # 0 -> ignore (unlabeled)
}


def load_scannet_raw_id_to_unified(tsv_path):
    """
    Build a RAW ScanNet category id -> UNIFIED_CLASSES mapping via the two-step
    composition: raw 'id' -> 'nyu40id' -> unified, using ScanNet's official
    scannetv2-labels.combined.tsv. This is what you need if your label/ PNGs
    store the raw ~607-category id (common in several third-party 2D ScanNet
    repackagings), NOT already-reduced nyu40id values directly -- raw id and
    nyu40id are DIFFERENT numbering systems (e.g. id=2 "chair" -> nyu40id=5),
    so applying NYU40_ID_TO_UNIFIED straight to raw ids silently mismaps almost
    everything (any raw id > 40 becomes ignore_index outright, and the ones
    that happen to fall in 1-40 map to the WRONG class by coincidence).
    """
    id_to_nyu40 = {}
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        id_idx = header.index("id")
        nyu40_idx = header.index("nyu40id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(id_idx, nyu40_idx):
                continue
            try:
                raw_id = int(parts[id_idx])
                nyu40id = int(parts[nyu40_idx])
            except ValueError:
                continue
            id_to_nyu40[raw_id] = nyu40id

    mapping = {}
    for raw_id, nyu40id in id_to_nyu40.items():
        if nyu40id in NYU40_ID_TO_UNIFIED:   # nyu40id==0 (void) is correctly left unmapped -> ignore
            mapping[raw_id] = NYU40_ID_TO_UNIFIED[nyu40id]
    return mapping

# Fallback constant Cityscapes stereo calibration (used only if camera.json is
# missing). Real per-image values are more accurate -- download
# camera_trainvaltest.zip if you want this to be exact.
DEFAULT_CS_BASELINE = 0.209313
DEFAULT_CS_FX = 2262.52


def remap_labels(label_img, mapping, ignore_index=IGNORE_INDEX):
    out = np.full(label_img.shape, ignore_index, dtype=np.uint8)
    for src_id, dst_id in mapping.items():
        out[label_img == src_id] = dst_id
    return out


def resize_triplet(img, seg, depth, target_short_side):
    h, w = img.shape[:2]
    scale = target_short_side / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    seg_r = cv2.resize(seg, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # NOTE: nearest-neighbor is used for depth too, to avoid blending valid
    # depth values into invalid (0) border/sky regions during downsampling.
    depth_r = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_r, seg_r, depth_r


def save_sample(out_split_dir, name, rgb, seg_unified, depth_m, depth_scale=1000.0):
    os.makedirs(os.path.join(out_split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_split_dir, "seg"), exist_ok=True)
    os.makedirs(os.path.join(out_split_dir, "depth"), exist_ok=True)
    cv2.imwrite(os.path.join(out_split_dir, "images", f"{name}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_split_dir, "seg", f"{name}.png"), seg_unified)
    depth_mm = np.clip(depth_m * depth_scale, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_split_dir, "depth", f"{name}.png"), depth_mm)


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------
def disparity_to_depth(disp_png, baseline, fx):
    disp = disp_png.astype(np.float32)
    valid = disp > 0
    disp = np.where(valid, (disp - 1) / 256.0, 0)   # Cityscapes disparity encoding
    depth = np.zeros_like(disp)
    depth[valid] = (baseline * fx) / np.maximum(disp[valid], 1e-6)
    return depth  # meters


def process_cityscapes(gtfine_root, img_root, disp_root, camera_root, out_dir,
                        target_short_side, domain_lines_train, domain_lines_val,
                        max_depth=80.0):
    n_written = {"train": 0, "val": 0}
    n_real_calib = 0    # how many images actually found a per-image camera.json
    n_fallback_calib = 0  # how many silently used the constant DEFAULT_CS_BASELINE/FX
    n_frames_skipped_unreadable = 0  # file existed but cv2 couldn't decode it
    for split, out_split in [("train", "train"), ("val", "val")]:
        img_split_dir = os.path.join(img_root, "leftImg8bit", split)
        if not os.path.isdir(img_split_dir):
            print(f"[Cityscapes] {img_split_dir} not found, skipping split '{split}'.")
            continue
        for city in sorted(os.listdir(img_split_dir)):
            city_img_dir = os.path.join(img_split_dir, city)
            for fname in sorted(os.listdir(city_img_dir)):
                if not fname.endswith("_leftImg8bit.png"):
                    continue
                stem = fname.replace("_leftImg8bit.png", "")
                img_path = os.path.join(city_img_dir, fname)
                seg_path = os.path.join(gtfine_root, "gtFine", split, city, f"{stem}_gtFine_labelIds.png")
                disp_path = os.path.join(disp_root, "disparity", split, city, f"{stem}_disparity.png")
                if not (os.path.isfile(seg_path) and os.path.isfile(disp_path)):
                    continue

                baseline, fx = DEFAULT_CS_BASELINE, DEFAULT_CS_FX
                used_real_calib = False
                if camera_root:
                    cam_path = os.path.join(camera_root, "camera", split, city, f"{stem}_camera.json")
                    if os.path.isfile(cam_path):
                        with open(cam_path) as f:
                            cam = json.load(f)
                        baseline = cam["extrinsic"]["baseline"]
                        fx = cam["intrinsic"]["fx"]
                        used_real_calib = True
                if used_real_calib:
                    n_real_calib += 1
                else:
                    n_fallback_calib += 1

                img_bgr = cv2.imread(img_path)
                seg_raw = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
                disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
                if img_bgr is None or seg_raw is None or disp is None:
                    n_frames_skipped_unreadable += 1
                    continue
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                seg_unified = remap_labels(seg_raw, CITYSCAPES_RAW_ID_TO_UNIFIED)
                depth = disparity_to_depth(disp, baseline, fx)
                depth = np.clip(depth, 0, max_depth)

                img, seg_unified, depth = resize_triplet(img, seg_unified, depth, target_short_side)

                name = f"city_{city}_{stem}"
                out_split_dir = os.path.join(out_dir, out_split)
                save_sample(out_split_dir, name, img, seg_unified, depth)
                (domain_lines_val if out_split == "val" else domain_lines_train).append(f"{name} 1")
                n_written[out_split] += 1
    print(f"[Cityscapes] wrote {n_written['train']} train / {n_written['val']} val samples")
    if n_frames_skipped_unreadable > 0:
        print(f"[Cityscapes] WARNING: {n_frames_skipped_unreadable} images were skipped because a "
              f"file existed but couldn't be decoded (corrupted/truncated/0-byte).")
    if n_fallback_calib > 0:
        print(f"[Cityscapes] WARNING: {n_fallback_calib}/{n_real_calib + n_fallback_calib} images used the "
              f"FALLBACK constant calibration (baseline={DEFAULT_CS_BASELINE}, fx={DEFAULT_CS_FX}) instead of "
              f"their real per-image camera.json -- this introduces a systematic depth error on those images. "
              f"{'Pass --cityscapes_camera pointing at the extracted camera_trainvaltest.zip to fix this.' if n_real_calib == 0 else ''}")
    else:
        print(f"[Cityscapes] all {n_real_calib} images used real per-image calibration from camera.json. Good.")


# ---------------------------------------------------------------------------
# ScanNet
# ---------------------------------------------------------------------------
def process_scannet(root, out_dir, target_short_side, val_ratio, seed,
                     domain_lines_train, domain_lines_val, label_map, max_depth=10.0):
    scenes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val_scenes = max(1, int(len(scenes) * val_ratio))
    val_scenes = set(scenes[:n_val_scenes])   # split by SCENE, not by frame -> no leakage

    n_written = {"train": 0, "val": 0}
    n_scenes_skipped = 0
    n_frames_skipped_unreadable = 0   # file existed but cv2 couldn't decode it (corrupt/truncated/0-byte)
    for scene in scenes:
        color_dir = os.path.join(root, scene, "color")
        depth_dir = os.path.join(root, scene, "depth")
        label_dir = os.path.join(root, scene, "label")
        if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir) and os.path.isdir(label_dir)):
            print(f"[ScanNet] {scene}: missing color/depth/label folder, skipping scene")
            n_scenes_skipped += 1
            continue
        out_split = "val" if scene in val_scenes else "train"
        out_split_dir = os.path.join(out_dir, out_split)

        color_files = sorted(os.listdir(color_dir))
        for fname in color_files:
            stem = os.path.splitext(fname)[0]
            color_path = os.path.join(color_dir, fname)
            depth_path = None
            label_path = None
            for ext in (".png", ".jpg"):
                if os.path.isfile(os.path.join(depth_dir, stem + ext)):
                    depth_path = os.path.join(depth_dir, stem + ext)
                if os.path.isfile(os.path.join(label_dir, stem + ext)):
                    label_path = os.path.join(label_dir, stem + ext)
            if depth_path is None or label_path is None:
                continue

            # Robustness: any of these three can come back None if the file exists on
            # disk but is corrupted/truncated/0-byte (e.g. from an interrupted download
            # or a disk-full event during extraction) -- cv2.imread returns None rather
            # than raising, and the ORIGINAL code didn't check, so one bad frame anywhere
            # in the whole dataset would crash the entire multi-hour run. Skip + count
            # instead, so processing completes and you get an honest total at the end.
            img_bgr = cv2.imread(color_path)
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            label_raw = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            if img_bgr is None or depth_raw is None or label_raw is None:
                n_frames_skipped_unreadable += 1
                continue

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            depth_raw = depth_raw.astype(np.float32)
            depth_m = depth_raw / 1000.0            # ScanNet depth pngs are millimeters
            depth_m = np.clip(depth_m, 0, max_depth)
            if label_raw.ndim == 3:
                label_raw = label_raw[..., 0]
            seg_unified = remap_labels(label_raw, label_map)

            # depth/label are typically already ScanNet's native (often lower)
            # resolution than color; resize label+depth onto the color frame first
            if depth_m.shape[:2] != img.shape[:2]:
                depth_m = cv2.resize(depth_m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if seg_unified.shape[:2] != img.shape[:2]:
                seg_unified = cv2.resize(seg_unified, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            img, seg_unified, depth_m = resize_triplet(img, seg_unified, depth_m, target_short_side)

            name = f"scan_{scene}_{stem}"
            save_sample(out_split_dir, name, img, seg_unified, depth_m)
            (domain_lines_val if out_split == "val" else domain_lines_train).append(f"{name} 0")
            n_written[out_split] += 1

    print(f"[ScanNet] wrote {n_written['train']} train / {n_written['val']} val samples "
          f"({len(scenes) - len(val_scenes)} train scenes / {len(val_scenes)} val scenes)")
    if n_scenes_skipped > 0:
        print(f"[ScanNet] WARNING: {n_scenes_skipped}/{len(scenes)} scenes were skipped entirely "
              f"(missing color/depth/label folder). If this is a large fraction of your scenes, "
              f"your ScanNet download/extraction is likely incomplete -- check for a disk-full or "
              f"interrupted extraction, and consider re-extracting the archive.")
    if n_frames_skipped_unreadable > 0:
        print(f"[ScanNet] WARNING: {n_frames_skipped_unreadable} individual frames were skipped "
              f"because a color/depth/label file existed but couldn't be decoded (corrupted/truncated/"
              f"0-byte). Same likely cause as above -- worth re-extracting if this number is large.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityscapes_gtfine", default="", help="path containing gtFine/ (i.e. extracted gtFine_trainvaltest)")
    ap.add_argument("--cityscapes_img", default="", help="path containing leftImg8bit/ (i.e. extracted leftImg8bit_trainvaltest)")
    ap.add_argument("--cityscapes_disp", default="", help="path containing disparity/ (i.e. extracted disparity_trainvaltest)")
    ap.add_argument("--cityscapes_camera", default="", help="optional: path containing camera/ (extracted camera_trainvaltest)")
    ap.add_argument("--scannet_root", default="", help="path to scannet_2d/ (contains scene####_##/ folders)")
    ap.add_argument("--scannet_label_convention", default="raw_id", choices=["raw_id", "nyu40"],
                     help="'raw_id' (default, safer): label/ PNGs store ScanNet's raw ~607-category id, "
                          "which needs --scannet_label_map_tsv to translate to nyu40/unified classes. "
                          "'nyu40': label/ PNGs ALREADY store nyu40id (0-40) directly, e.g. if you're using "
                          "the preprocessed scannet_frames_25k release -- no tsv needed in that case. "
                          "Run data/inspect_dataset.py first if unsure: raw label PNG values noticeably "
                          "above 40 mean you need 'raw_id' (the default).")
    ap.add_argument("--scannet_label_map_tsv", default="",
                     help="path to scannetv2-labels.combined.tsv (required when "
                          "--scannet_label_convention=raw_id, i.e. the default). Download from e.g. "
                          "https://raw.githubusercontent.com/facebookresearch/votenet/main/scannet/"
                          "meta_data/scannetv2-labels.combined.tsv")
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--target_short_side", type=int, default=480,
                     help="both domains are resized so their shorter side equals this, "
                          "preserving aspect ratio -- keeps Cityscapes' 2048x1024 and "
                          "ScanNet's 320x240 on a comparable scale for training")
    ap.add_argument("--scannet_val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    domain_lines_train, domain_lines_val = [], []

    if args.cityscapes_gtfine and args.cityscapes_img and args.cityscapes_disp:
        process_cityscapes(args.cityscapes_gtfine, args.cityscapes_img, args.cityscapes_disp,
                            args.cityscapes_camera, args.out, args.target_short_side,
                            domain_lines_train, domain_lines_val)
    else:
        print("[Cityscapes] skipped -- need --cityscapes_gtfine, --cityscapes_img, --cityscapes_disp")

    if args.scannet_root:
        if args.scannet_label_convention == "raw_id":
            if not args.scannet_label_map_tsv:
                raise SystemExit(
                    "--scannet_label_convention=raw_id (the default) requires --scannet_label_map_tsv "
                    "pointing at scannetv2-labels.combined.tsv. If your label/ PNGs actually already store "
                    "nyu40id (0-40) directly -- confirm with data/inspect_dataset.py first -- pass "
                    "--scannet_label_convention nyu40 instead and omit this flag."
                )
            label_map = load_scannet_raw_id_to_unified(args.scannet_label_map_tsv)
            print(f"[ScanNet] using raw-id->nyu40->unified mapping from {args.scannet_label_map_tsv} "
                  f"({len(label_map)} raw ids mapped)")
        else:
            label_map = NYU40_ID_TO_UNIFIED
            print("[ScanNet] using direct nyu40id->unified mapping (label PNGs assumed to already store nyu40id)")
        process_scannet(args.scannet_root, args.out, args.target_short_side, args.scannet_val_ratio,
                         args.seed, domain_lines_train, domain_lines_val, label_map)
    else:
        print("[ScanNet] skipped -- need --scannet_root")

    for split, lines in [("train", domain_lines_train), ("val", domain_lines_val)]:
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, "domain.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump({"classes": UNIFIED_CLASSES, "ignore_index": IGNORE_INDEX}, f, indent=2)

    print(f"\nDone. {len(domain_lines_train)} train / {len(domain_lines_val)} val samples written to {args.out}")


if __name__ == "__main__":
    main()
