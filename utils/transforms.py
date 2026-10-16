"""
Preprocessing + augmentation shared by segmentation and depth targets so that
geometric transforms (flip, crop, resize) are applied IDENTICALLY to image,
segmentation mask, and depth map. Photometric transforms (color jitter, blur,
noise) are applied to the image only.
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(img_size: int, mean, std, ignore_index: int = 255):
    return A.Compose(
        [
            # ---- geometric augmentation (applied identically to img/seg/depth) ----
            A.RandomScale(scale_limit=0.2, p=0.5),
            # fill_mask=ignore_index (NOT 0): 0 is a REAL segmentation class
            # ("floor_ground" in UNIFIED_CLASSES), so filling padding with 0 would
            # teach the model that black borders are valid floor pixels. ignore_index
            # (255) excludes these synthetic border pixels from the seg loss entirely.
            # It also happens to correctly invalidate the "depth" additional target at
            # these pixels (255m > max_depth for either domain -> excluded by the
            # depth valid-mask too), so one fill value correctly handles both targets.
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=ignore_index),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=ignore_index, p=0.3),
            # ---- photometric augmentation / enhancement (image only) ----
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),                       # contrast enhancement
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
            A.ImageCompression(quality_range=(70, 100), p=0.1),
            # ---- normalization ----
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        additional_targets={"depth": "mask"},   # depth resized/cropped with nearest-safe ops like a mask
    )


def build_train_transforms_strong(img_size: int, mean, std, ignore_index: int = 255):
    """
    Stronger augmentation pipeline for the MINORITY domain under balanced sampling.
    With balanced_domain_sampling=True, WeightedRandomSampler revisits the smaller
    domain's images many times per epoch (e.g. ~13.5x for outdoor here) -- those are
    literally the same underlying files repeated, which risks the model memorizing
    them rather than generalizing. Using a heavier augmentation pipeline for just
    that domain makes each repeat visit look meaningfully different, which is a
    standard mitigation for oversampling-induced overfitting on a minority class.
    """
    return A.Compose(
        [
            A.RandomScale(scale_limit=0.3, p=0.6),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=ignore_index),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=ignore_index, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            A.CLAHE(clip_limit=3.0, p=0.25),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.12), p=0.2),
            A.ImageCompression(quality_range=(55, 100), p=0.2),
            # fill_mask=ignore_index so dropped-out regions are excluded from both the
            # segmentation loss AND the depth valid-mask (255m > max_depth for either
            # domain), rather than leaving a mismatched image/label pair in those pixels.
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.10),
                             hole_width_range=(0.03, 0.10), fill=0, fill_mask=ignore_index, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        additional_targets={"depth": "mask"},
    )


def build_val_transforms(img_size: int, mean, std):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        additional_targets={"depth": "mask"},
    )
