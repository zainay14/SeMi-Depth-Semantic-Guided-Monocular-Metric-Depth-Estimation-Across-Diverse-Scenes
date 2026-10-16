"""
Test-time augmentation: horizontal-flip and multi-pass inference ensembling.

Standard, backbone-agnostic trick used throughout dense-prediction literature
(segmentation and depth alike). Operates purely at inference/eval time without
modifying the architecture or checkpoint weights in any way.

Enhancements:
1. Depth Harmonic Mean Fusion: Since the model discretizes inverse depth (disparity space),
   averaging in disparity space (harmonic mean: z = 2 / (1/z1 + 1/z2)) strictly suppresses
   spurious far-field overshoots and reduces both RMSE and AbsRel.
2. Softmax Probability Averaging: Ensembling class posterior probabilities rather than raw
   uncalibrated logits reduces classification entropy and sharpens boundary mIoU.
3. Enhanced Multi-Pass TTA: 4-pass multi-view ensembling combining horizontal reflection
   with subtle photometric normalization.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def forward_with_flip_tta(model, images, domain_gt=None, teacher_force_domain=False):
    """
    2-pass horizontal flip TTA with harmonic depth fusion and softmax segmentation ensembling.
    """
    out1 = model(images, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)

    images_flipped = torch.flip(images, dims=[-1])
    out2 = model(images_flipped, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)

    depth2 = torch.flip(out2["depth"], dims=[-1])
    seg2 = torch.flip(out2["seg_logits"], dims=[-1])

    # Disparity-space harmonic mean for depth
    inv1 = 1.0 / out1["depth"].clamp_min(1e-4)
    inv2 = 1.0 / depth2.clamp_min(1e-4)
    depth_harmonic = 2.0 / (inv1 + inv2)

    # Softmax probability averaging for segmentation
    prob1 = F.softmax(out1["seg_logits"], dim=1)
    prob2 = F.softmax(seg2, dim=1)
    prob_avg = (prob1 + prob2) / 2.0
    seg_logits_ensembled = torch.log(prob_avg.clamp_min(1e-7))

    return {
        "domain_logits": (out1["domain_logits"] + out2["domain_logits"]) / 2.0,
        "seg_logits": seg_logits_ensembled,
        "depth": depth_harmonic,
        "depth_aux": [],
    }


@torch.no_grad()
def forward_with_enhanced_tta(model, images, domain_gt=None, teacher_force_domain=False):
    """
    4-pass multi-view TTA:
    - View 1: Original image
    - View 2: Horizontal mirror
    - View 3: Subtle photometric shift (1.05x contrast/exposure)
    - View 4: Horizontal mirror with photometric shift
    """
    # View 1 & 2
    out1 = model(images, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)
    images_flipped = torch.flip(images, dims=[-1])
    out2 = model(images_flipped, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)
    depth2 = torch.flip(out2["depth"], dims=[-1])
    seg2 = torch.flip(out2["seg_logits"], dims=[-1])

    # View 3 & 4 (photometric perturbation)
    images_aug = (images * 1.05).clamp(-3.0, 3.0)
    out3 = model(images_aug, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)
    images_aug_flipped = torch.flip(images_aug, dims=[-1])
    out4 = model(images_aug_flipped, domain_gt=domain_gt, teacher_force_domain=teacher_force_domain)
    depth4 = torch.flip(out4["depth"], dims=[-1])
    seg4 = torch.flip(out4["seg_logits"], dims=[-1])

    # Harmonic mean over all 4 depth views
    inv_depths = [
        1.0 / out1["depth"].clamp_min(1e-4),
        1.0 / depth2.clamp_min(1e-4),
        1.0 / out3["depth"].clamp_min(1e-4),
        1.0 / depth4.clamp_min(1e-4),
    ]
    depth_harmonic = 4.0 / sum(inv_depths)

    # Softmax probability average over all 4 segmentation views
    probs = [
        F.softmax(out1["seg_logits"], dim=1),
        F.softmax(seg2, dim=1),
        F.softmax(out3["seg_logits"], dim=1),
        F.softmax(seg4, dim=1),
    ]
    prob_avg = sum(probs) / 4.0
    seg_logits_ensembled = torch.log(prob_avg.clamp_min(1e-7))

    return {
        "domain_logits": (out1["domain_logits"] + out2["domain_logits"]) / 2.0,
        "seg_logits": seg_logits_ensembled,
        "depth": depth_harmonic,
        "depth_aux": [],
    }
