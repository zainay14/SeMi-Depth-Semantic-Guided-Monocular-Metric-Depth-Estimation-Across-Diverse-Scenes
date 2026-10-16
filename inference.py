"""
Run the full pipeline on a single image and save visualizations.

Usage:
    python inference.py --ckpt checkpoints/best.pt --image path/to/img.jpg --out ./pred
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import data_cfg, model_cfg
from models.full_model import MultiTaskModel

DOMAIN_NAMES = ["indoor", "outdoor"]


def load_image(path, img_size, mean, std):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    orig = img.copy()
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - np.array(mean)) / np.array(std)
    tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, orig


def colorize_seg(seg_map, num_classes):
    rng = np.random.RandomState(42)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[255 % num_classes] = [0, 0, 0]
    return palette[seg_map % num_classes]


def colorize_depth(depth_map):
    d = depth_map.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_MAGMA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="./pred")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskModel(model_cfg, data_cfg.num_seg_classes, img_size=data_cfg.img_size).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tensor, orig = load_image(args.image, data_cfg.img_size, data_cfg.mean, data_cfg.std)
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor, domain_gt=None, teacher_force_domain=False)

    domain_pred = outputs["domain_logits"].softmax(1).argmax(1).item()
    seg_pred = outputs["seg_logits"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    depth_pred = outputs["depth"].squeeze(0).cpu().numpy()

    seg_color = colorize_seg(seg_pred, data_cfg.num_seg_classes)
    seg_color = cv2.resize(seg_color, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth_color = colorize_depth(depth_pred)
    depth_color = cv2.resize(depth_color, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

    base = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{base}_seg.png"), cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out, f"{base}_depth.png"), depth_color)
    np.save(os.path.join(args.out, f"{base}_depth_meters.npy"), depth_pred)

    print(f"Predicted domain: {DOMAIN_NAMES[domain_pred]}")
    print(f"Saved: {base}_seg.png, {base}_depth.png, {base}_depth_meters.npy in {args.out}")


if __name__ == "__main__":
    main()
