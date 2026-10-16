"""
Loss functions for all four branches.

    L_total = alpha * L_domain + beta * L_segmentation + gamma * L_depth + delta * L_boundary
              + edge_smooth_weight * L_smooth

- L_domain:    cross-entropy (indoor/outdoor)
- L_segmentation: cross-entropy with label smoothing + Dice (helps class imbalance)
- L_depth:     per-domain weighted combination of Scale-Invariant Log loss (Eigen et al.)
               + L1(log) + a direct linear-space MSE term, valid-pixel masked. Computed
               SEPARATELY for indoor/outdoor subsets of the batch and recombined with
               loss_cfg.depth_weight_indoor / depth_weight_outdoor so one domain's error
               scale can't silently dominate or be starved relative to the other.
- L_smooth:    edge-aware smoothness on log-depth, gated by image gradients (standard
               monocular-depth regularizer; discourages noisy high-frequency error in
               large flat regions like road/sky/wall, which otherwise inflate RMSE).
- L_boundary:  Sobel-gradient consistency between predicted segmentation boundaries
               and predicted depth discontinuities (Boundary Consistency Loss)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
def domain_loss(domain_logits, domain_gt):
    return F.cross_entropy(domain_logits, domain_gt)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def dice_loss(logits, target, num_classes, ignore_index, eps=1e-6):
    valid = (target != ignore_index)
    target_clamped = target.clone()
    target_clamped[~valid] = 0
    target_onehot = F.one_hot(target_clamped, num_classes).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)

    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    target_onehot = target_onehot * valid_f

    dims = (0, 2, 3)
    intersection = torch.sum(probs * target_onehot, dims)
    union = torch.sum(probs + target_onehot, dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss(log_probs, seg_gt, ignore_index, gamma=2.0, weight=None):
    """
    Focal Loss (Lin et al., CVPR 2017).

    Multiplies the standard per-pixel NLL by (1-p_t)^gamma, where p_t is the
    model's probability on the correct class. Well-classified, easy pixels have
    large p_t -> small focal weight -> small gradient contribution. Hard/rare
    pixels (ceiling, person with 0.0 IoU) have small p_t -> focal weight near 1
    -> gradient contribution unchanged. Net effect: the model stops wasting
    capacity on the already-correct majority and focuses on the hard minority.

    gamma=2.0 is the original paper's default. weight (B,H,W) can pass an
    existing per-pixel weight tensor (e.g. class or boundary weight) so focal
    and existing weighting compose multiplicatively.
    """
    valid = (seg_gt != ignore_index)
    if valid.sum() == 0:
        return log_probs.sum() * 0.0
    seg_gt_clamped = seg_gt.clone()
    seg_gt_clamped[~valid] = 0
    # NLL per pixel: shape (B, H, W)
    nll = -log_probs.gather(1, seg_gt_clamped.unsqueeze(1)).squeeze(1)
    p_t = torch.exp(-nll).clamp(min=1e-6)
    focal_w = (1.0 - p_t) ** gamma          # (B, H, W)
    per_pixel = focal_w * nll
    if weight is not None:
        denom = weight[valid].sum().clamp_min(1e-6)
        return (per_pixel[valid] * weight[valid]).sum() / denom
    return per_pixel[valid].mean()


# ---------------------------------------------------------------------------
# Lovász-Softmax Loss (Berman et al., CVPR 2018) -- inline to avoid pip deps
# Directly minimises 1 - mIoU. CE and Dice both optimise proxy objectives;
# Lovász optimises the exact Jaccard index that mIoU measures. It is the
# gold-standard fix when per-class IoU is low.
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted):
    """Lovász extension gradient for a single class (see Algorithm 1, Berman 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, ignore_index=255):
    """
    Multi-class Lovász-Softmax loss.

    probas: (B, C, H, W)  -- softmax probabilities (NOT log-probs)
    labels: (B, H, W)     -- integer ground-truth class indices
    Returns a scalar loss that, when minimised, directly maximises mean IoU.
    """
    B, C, H, W = probas.shape
    probas_flat = probas.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
    labels_flat = labels.reshape(-1)                            # (N,)
    valid = labels_flat != ignore_index
    if valid.sum() == 0:
        return probas.sum() * 0.0

    pv = probas_flat[valid]    # (M, C)
    lv = labels_flat[valid]    # (M,)

    losses = []
    for c in range(C):
        fg = (lv == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - pv[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def seg_boundary_distance_weight(seg_gt, ignore_index=255, near_distance=5, power=1.5, min_weight=0.3, max_iters=20):
    """
    Boundary-distance reliability weight for segmentation, mirroring the same
    clip/power/floor shape as depth_reliability_weight -- but using distance from the
    nearest LABEL BOUNDARY instead of distance from the camera. Rationale: unlike
    depth noise (caused by sensor physics), segmentation label noise mainly comes from
    annotation ambiguity -- pixels right at a class transition (e.g. a car's silhouette
    against the road) are inherently less certain than pixels deep inside a large,
    unambiguous region (open road, sky). This is a well-documented property of
    polygon-based semantic annotation generally, not specific to this dataset.

    Computed entirely from the GROUND-TRUTH label map (not model predictions or
    learned), matching the interpretability property of the depth-side weighting --
    "this pixel matters less because it's near an annotation boundary" is just as
    defensible as "this pixel matters less because it's far away."

    Implementation note: a proper Euclidean distance transform (e.g. scipy) would
    require moving this computation into the CPU-side data pipeline. To keep this a
    pure loss-side change (consistent with every other robustness fix in this
    project), this instead computes a GPU-native discrete (chessboard) distance
    approximation via iterative max-pool dilation of the boundary mask: boundary
    pixels start at distance 0, each dilation round marks newly-covered pixels with
    the current round number, up to max_iters (pixels never reached are treated as
    "far enough" to receive the full-trust ceiling, same as depth pixels well past
    the reliability floor). This is a coarser distance metric than a true Euclidean
    transform but is monotonically correct (farther-from-boundary pixels always get
    a distance >= closer pixels'), which is all the weighting formula needs.

    Returns a per-pixel weight the same shape as seg_gt, in [min_weight, 1].
    """
    B, H, W = seg_gt.shape
    device = seg_gt.device
    valid = (seg_gt != ignore_index)

    # A pixel is a boundary pixel if it differs from any of its 4-connected
    # neighbours (or borders an ignored region) -- computed via simple shifted
    # comparisons, fully GPU-vectorized, no scipy/CPU round-trip needed.
    padded = F.pad(seg_gt.float().unsqueeze(1), (1, 1, 1, 1), mode="replicate")  # (B,1,H+2,W+2)
    center = padded[:, :, 1:-1, 1:-1]
    up = padded[:, :, :-2, 1:-1]
    down = padded[:, :, 2:, 1:-1]
    left = padded[:, :, 1:-1, :-2]
    right = padded[:, :, 1:-1, 2:]
    is_boundary = ((center != up) | (center != down) | (center != left) | (center != right)).squeeze(1)  # (B,H,W)

    dist = torch.full((B, H, W), float(max_iters), device=device)
    dist[is_boundary] = 0.0
    frontier = is_boundary.float().unsqueeze(1)  # (B,1,H,W)
    for step in range(1, max_iters + 1):
        dilated = F.max_pool2d(frontier, kernel_size=3, stride=1, padding=1)
        newly_covered = (dilated[:, 0] > 0) & (dist == float(max_iters))
        dist[newly_covered] = float(step)
        frontier = dilated
        if newly_covered.sum() == 0:
            break

    w = torch.clamp((dist / near_distance).clamp(min=1e-6) ** (1.0 / power), max=1.0)
    # ^ inverse-shaped clip: distance >= near_distance -> full trust (1.0); distance 0
    #   (right on a boundary) -> approaches 0, floored at min_weight below.
    w = torch.clamp(w, min=min_weight, max=1.0)
    w = torch.where(valid, w, torch.zeros_like(w))
    return w


def segmentation_loss(seg_logits, seg_gt, num_classes, ignore_index=255, label_smoothing=0.05,
                       robust_weight=0.2, robust_q=0.7, boundary_weighting=True,
                       boundary_near=5, boundary_power=1.5, boundary_min_weight=0.3,
                       class_weights=None, focal_weight=0.4, focal_gamma=2.0,
                       lovasz_weight=0.4):
    """
    Manual masked, label-smoothed cross-entropy + Dice, instead of relying on
    F.cross_entropy(ignore_index=..., label_smoothing=...) directly.

    Why: F.cross_entropy's mean reduction divides by the count of non-ignored
    elements ACROSS THE WHOLE BATCH. If every pixel in a batch happens to be
    ignore_index -- e.g. one real source image with a fully void/unlabeled seg map,
    landing alone in a small batch -- that's 0/0 = NaN, regardless of label_smoothing
    (verified directly: F.cross_entropy on an all-ignored batch returns NaN even with
    label_smoothing=0.0). This computes the identical label-smoothed CE formula by
    hand from log_softmax, masks BEFORE averaging, and explicitly returns 0 for the
    (should be rare, but real) all-invalid-batch case instead of ever dividing by zero.

    robust_weight (blended with standard CE, not a replacement): standard cross-entropy
    -log(p) is UNBOUNDED as p->0 -- if the model becomes confident about a pixel that
    happens to carry a wrong/noisy label (annotation error, boundary ambiguity, any
    residual label noise), that single pixel's loss can explode and dominate the
    batch's gradient, the same failure mode the depth-side outlier fixes address.
    Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018) -- L_q(p) = (1 - p^q) / q
    -- is the standard, well-cited fix from the noisy-label literature: bounded (max
    value 1/q, unlike CE's unbounded range), while still reducing to something
    CE-proportional for well-classified pixels, so clean/correct pixels keep training
    normally. Blended at a modest weight (not a full replacement) so it dampens
    outlier-pixel domination without meaningfully changing convergence on the clean
    majority of labels. q=0.7 is the value recommended in the original paper. Set
    robust_weight=0.0 to disable and use pure CE+Dice as before.

    boundary_weighting: additionally weights the cross-entropy term by
    seg_boundary_distance_weight (see its docstring) -- pixels near an annotation
    boundary contribute less than pixels deep inside a confidently-labelled region.
    This is complementary to robust_weight/GCE: GCE bounds how much a single
    surprising pixel can dominate the gradient AFTER the fact (based on the model's
    own confidence); boundary weighting reduces trust BEFOREHAND, based only on the
    ground-truth label geometry, independent of what the model predicts.

    class_weights (optional, tensor of shape (num_classes,) or None): per-class
    inverse-frequency weight, see compute_class_weights.py. Addresses a DIFFERENT
    failure mode than boundary_weighting or GCE -- both of those adjust trust based
    on where/how confident a prediction is, but neither helps a class that's simply
    RARE across the whole dataset (e.g. this project's own eval showed ceiling,
    indoor "person", and outdoor "window" at exactly 0.0 IoU -- a rare-class problem,
    not a boundary-noise or label-confidence problem). Applied multiplicatively with
    the boundary weight, not as a replacement -- a rare-class pixel that's ALSO deep
    inside a confident region should count more than a rare-class pixel right at a
    fuzzy boundary. Pass None to disable (matches prior behaviour exactly).
    """
    valid = (seg_gt != ignore_index)
    if valid.sum() == 0:
        return seg_logits.sum() * 0.0

    log_probs = F.log_softmax(seg_logits, dim=1)                              # (B,C,H,W)
    seg_gt_clamped = seg_gt.clone()
    seg_gt_clamped[~valid] = 0
    nll = -log_probs.gather(1, seg_gt_clamped.unsqueeze(1)).squeeze(1)        # (B,H,W) hard-label term
    smooth = -log_probs.mean(dim=1)                                          # (B,H,W) uniform-smoothing term
    per_pixel = (1 - label_smoothing) * nll + label_smoothing * smooth

    # Combine boundary weight and class weight multiplicatively into one per-pixel
    # weight tensor, so both mechanisms (if enabled) apply consistently to every
    # term below (CE, GCE) rather than each needing its own separate weighting path.
    combined_weight = None
    if boundary_weighting:
        combined_weight = seg_boundary_distance_weight(seg_gt, ignore_index=ignore_index, near_distance=boundary_near,
                                                         power=boundary_power, min_weight=boundary_min_weight)
    if class_weights is not None:
        cw_per_pixel = class_weights.to(seg_gt.device)[seg_gt_clamped]  # (B,H,W), gather by GT class
        combined_weight = cw_per_pixel if combined_weight is None else combined_weight * cw_per_pixel

    if combined_weight is not None:
        ce = _weighted_mean(per_pixel[valid], combined_weight[valid])
    else:
        ce = per_pixel[valid].mean()

    if robust_weight > 0:
        p_true = torch.exp(-nll).clamp(min=1e-6, max=1.0)                     # p(true class), from the same log_probs
        gce = (1.0 - p_true ** robust_q) / robust_q                           # bounded in [0, 1/q]
        if combined_weight is not None:
            gce_term = _weighted_mean(gce[valid], combined_weight[valid])
        else:
            gce_term = gce[valid].mean()
        ce = (1 - robust_weight) * ce + robust_weight * gce_term

    dl = dice_loss(seg_logits, seg_gt, num_classes, ignore_index)
    total_seg = ce + dl

    # Focal Loss: re-weight CE by (1-p_t)^gamma to focus on hard/rare classes
    if focal_weight > 0.0:
        fl = focal_loss(log_probs, seg_gt, ignore_index, gamma=focal_gamma,
                        weight=combined_weight)
        total_seg = total_seg + focal_weight * fl

    # Lovász-Softmax: directly minimise 1-mIoU (Berman et al., CVPR 2018)
    if lovasz_weight > 0.0:
        ls = lovasz_softmax(F.softmax(seg_logits, dim=1), seg_gt, ignore_index=ignore_index)
        total_seg = total_seg + lovasz_weight * ls

    return total_seg


# ---------------------------------------------------------------------------
# Depth: Scale-Invariant Log loss (Eigen, Puhrsch, Fergus 2014) + L1(log)
# ---------------------------------------------------------------------------
def _weighted_mean(per_pixel_loss, weight):
    denom = weight.sum().clamp_min(1e-6)
    return (per_pixel_loss * weight).sum() / denom


def silog_loss(pred_depth, gt_depth, valid_mask, lam=0.85, eps=1e-6, weight=None):
    pred = torch.clamp(pred_depth, min=eps)
    gt = torch.clamp(gt_depth, min=eps)
    diff_log = torch.log(pred) - torch.log(gt)
    diff_log = diff_log[valid_mask]
    if diff_log.numel() == 0:
        return pred_depth.sum() * 0.0
    if weight is None:
        n = diff_log.numel()
        term1 = torch.sum(diff_log ** 2) / n
        term2 = (torch.sum(diff_log) ** 2) / (n ** 2)
    else:
        # weighted generalization: weighted mean-of-squares and (weighted mean)^2 --
        # standard weighted-variance formula, reduces exactly to the unweighted case
        # above when weight is uniform.
        w = weight[valid_mask]
        wsum = w.sum().clamp_min(eps)
        term1 = torch.sum(w * diff_log ** 2) / wsum
        term2 = (torch.sum(w * diff_log) / wsum) ** 2
    # term1 - lam*term2 == Var(diff_log) + (1-lam)*mean(diff_log)^2, which is
    # ALWAYS >= 0 algebraically. But when a domain-subset has few valid pixels
    # (common now that depth_loss splits indoor/outdoor per-batch) and the model
    # is predicting near-perfectly for that subset, float32 can round this tiny
    # positive quantity to a tiny NEGATIVE one -- and sqrt(negative) = NaN. Clamp
    # to 0 first: the true value can never legitimately be negative, so this only
    # ever corrects float32 noise, never masks a real problem.
    variance_term = torch.clamp(term1 - lam * term2, min=0.0)
    return torch.sqrt(variance_term + eps) * 10.0   # x10 standard scaling for gradient magnitude


def depth_l1_loss(pred_depth, gt_depth, valid_mask, weight=None):
    if valid_mask.sum() == 0:
        return pred_depth.sum() * 0.0
    diff = torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask])
    if weight is None:
        return diff.mean()
    return _weighted_mean(diff, weight[valid_mask])


def berhu_loss(pred_depth, gt_depth, valid_mask, weight=None):
    """
    Reverse Huber (BerHu) loss (Laina et al. 2016, FCRN): L1 below an adaptive
    threshold c, L2 above it. Used in place of plain L1 specifically to push RMSE down --
    plain L1's gradient magnitude is constant regardless of error size, so it doesn't
    push harder on the large outlier errors that dominate RMSE (squared error). BerHu's
    L2 branch does exactly that for the worst errors, while staying L1 (robust, not
    over-sensitive to noisy GT) on the bulk of small errors.

    Threshold uses the 95th percentile of the batch's errors, NOT the max (the original
    Laina et al. formulation, and this codebase's original implementation). The max-based
    threshold has a real failure mode with noisy ground truth: a SINGLE extreme outlier
    pixel (very plausible with far-range Cityscapes disparity noise) balloons the
    threshold for the WHOLE batch, so genuinely-large-but-not-worst errors get
    misclassified as "small" and only receive the weak L1 gradient instead of the
    intended stronger L2 push -- the opposite of what BerHu is supposed to do. Verified
    directly: with one 30m outlier among 1000 normal (<0.5m) errors, the max-based
    threshold correctly penalized only 0.1% of real errors with L2; the percentile-based
    threshold correctly identifies 80.6%.
    """
    if valid_mask.sum() == 0:
        return pred_depth.sum() * 0.0
    diff = torch.abs(pred_depth[valid_mask] - gt_depth[valid_mask])
    c = 0.2 * torch.quantile(diff.detach(), 0.95).clamp(min=1e-3)
    l2_branch = (diff ** 2 + c ** 2) / (2 * c)
    out = torch.where(diff <= c, diff, l2_branch)
    if weight is None:
        return out.mean()
    return _weighted_mean(out, weight[valid_mask])


def depth_mse_loss(pred_depth, gt_depth, valid_mask, robust=False, robust_eps=1.0, weight=None):
    """
    Direct linear-space error term, masked. SILog/L1(log) optimize relative & log-scale
    error; neither directly minimizes RMSE (root mean SQUARED error, in meters). This
    term gives the RMSE metric a matching gradient.

    robust=False (default -- CHANGED from an earlier True default): plain MSE.
    Charbonnier (sqrt((pred-gt)^2+eps^2), the previous default) was tried specifically
    to bound outlier influence, but verified against two consecutive real checkpoints
    on this dataset: it did NOT improve outdoor RMSE, and mildly UNDERPERFORMED plain
    MSE on average. Kept available (robust=True) since it's not actively harmful, just
    not evidenced-helpful here -- the real fix for THIS dataset's specific noise
    pattern is the `weight` parameter below, not Charbonnier's outlier-detection
    approach. See depth_reliability_weight()'s docstring for why: this domain's noise
    is smoothly distance-correlated (predictable from GT directly), not random severe
    outliers (which is what Charbonnier targets).

    weight: optional per-pixel reliability weight (see depth_reliability_weight),
    applied as a weighted mean instead of plain mean.
    """
    if valid_mask.sum() == 0:
        return pred_depth.sum() * 0.0
    diff = pred_depth[valid_mask] - gt_depth[valid_mask]
    if robust:
        per_pixel = torch.sqrt(diff ** 2 + robust_eps ** 2) - robust_eps  # -eps so loss -> 0 as diff -> 0
    else:
        per_pixel = diff ** 2
    if weight is None:
        return per_pixel.mean()
    return _weighted_mean(per_pixel, weight[valid_mask])


def depth_reliability_weight(gt_depth, near_distance=15.0, power=2.0, min_weight=0.15):
    """
    Per-pixel confidence weight based on KNOWN ground-truth distance, not measured
    error -- a fundamentally different (and better-matched) signal than the BerHu/
    Charbonnier "was this pixel's error suspiciously large" approach.

    The physics: Cityscapes computes depth = baseline * fx / disparity. If disparity
    carries a roughly constant measurement uncertainty sigma_d (standard for stereo
    quantization noise), error propagation gives:
        sigma_depth = |d(depth)/d(disparity)| * sigma_d = (depth^2 / (baseline*fx)) * sigma_d
    i.e. depth uncertainty grows PROPORTIONAL TO depth^2 -- a pixel at 60m has ~36x the
    error scale of a pixel at 10m. This is a direct mathematical consequence of how
    your depth values were computed (verified against this project's own calibration
    pipeline earlier), not a guess.

    Why this beats outlier-detection losses (BerHu/Charbonnier) for THIS noise: those
    infer unreliability from a pixel's error AFTER the fact, which conflates "the GT is
    noisy here" with "the model made a genuine mistake here" -- they can't tell the two
    apart. This uses the GT distance directly, which we already know (independent of
    what the model predicts) correlates with reliability. Verified against real
    training data: two consecutive checkpoints showed Charbonnier NOT improving
    (mildly worsening) outdoor RMSE -- consistent with it being the wrong tool for
    smoothly distance-correlated noise rather than rare random outliers.

    weight = (near_distance / max(gt_depth, near_distance)) ^ power, floored at
    min_weight so no pixel is ever fully zeroed out -- every pixel still contributes
    real gradient signal, nothing is discarded (unlike outdoor_depth_train_max).
    power=2.0 (not the full theoretical 4.0 from the variance derivation above) is a
    deliberately gentler choice: the full inverse-VARIANCE weighting would suppress
    far pixels so heavily it's practically equivalent to excluding them, which
    contradicts the goal of keeping all data meaningfully contributing.
    """
    w = (near_distance / torch.clamp(gt_depth, min=near_distance)) ** power
    return torch.clamp(w, min=min_weight, max=1.0)


def chamfer_bin_loss(bin_centers, gt_depth, valid_mask, max_samples=1000, in_inv_depth=True):
    """
    1D Bidirectional Chamfer Loss between predicted bin centers and GT depths (AdaBins / SSRDepth).
    Operates in inverse-depth / disparity space when in_inv_depth=True to ensure uniform noise alignment.
    """
    if bin_centers is None or valid_mask.sum() == 0:
        return gt_depth.sum() * 0.0
    B = bin_centers.shape[0]
    total_chamfer = []
    for b in range(B):
        v = valid_mask[b]
        if v.sum() == 0:
            continue
        g = gt_depth[b][v]
        c = bin_centers[b]  # (N,)
        if in_inv_depth:
            g = 1.0 / g.clamp_min(1e-4)
            c = 1.0 / c.clamp_min(1e-4)
        if g.numel() > max_samples:
            idx = torch.randperm(g.numel(), device=g.device)[:max_samples]
            g = g[idx]
        dist = torch.abs(c.unsqueeze(1) - g.unsqueeze(0))  # (N, M)
        chamfer = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
        total_chamfer.append(chamfer)
    if not total_chamfer:
        return bin_centers.sum() * 0.0
    return torch.stack(total_chamfer).mean()


def _single_domain_depth_loss(pred_depth, gt_depth, valid_mask, loss_cfg, weight=None):
    recon_term = (berhu_loss(pred_depth, gt_depth, valid_mask, weight=weight) if loss_cfg.depth_use_berhu
                  else depth_l1_loss(pred_depth, gt_depth, valid_mask, weight=weight))
    loss = (silog_loss(pred_depth, gt_depth, valid_mask, lam=loss_cfg.depth_silog_lambda, weight=weight)
            + 0.5 * recon_term)
    if getattr(loss_cfg, "depth_mse_weight", 0.0) > 0:
        loss = loss + loss_cfg.depth_mse_weight * depth_mse_loss(pred_depth, gt_depth, valid_mask, weight=weight)
    return loss



def depth_class_importance_weight(seg_gt, important_classes=(9, 10), boost=1.5, ignore_index=255):
    """
    Semantic-class importance weight for depth loss -- complements (multiplies with)
    depth_reliability_weight rather than replacing it. Pure distance-based weighting
    treats a far pedestrian identically to a far patch of empty road, even though
    getting the pedestrian's depth wrong matters more in practice (safety-critical,
    dynamic objects) than a static background region. This uses the ALREADY-AVAILABLE
    ground-truth segmentation label to boost trust for pixels belonging to dynamic/
    safety-critical classes, independent of how far away they are.

    important_classes defaults to (9, 10) = (person, vehicle) in this project's
    15-class UNIFIED_CLASSES scheme (see data/prepare_dataset.py) -- the two classes
    covering people and all vehicle types (car/truck/bus/motorcycle/bicycle merged).
    boost=1.5 means these pixels get 1.5x the trust of a same-distance background
    pixel; ignore_index pixels get weight 0 (they're excluded from the loss elsewhere
    via valid_mask anyway, but zeroing here keeps this function self-consistent if
    ever used standalone).
    """
    w = torch.ones_like(seg_gt, dtype=torch.float32)
    for c in important_classes:
        w = torch.where(seg_gt == c, torch.full_like(w, boost), w)
    w = torch.where(seg_gt == ignore_index, torch.zeros_like(w), w)
    return w


def depth_loss(pred_depth, gt_depth, valid_mask, domain_gt, loss_cfg, seg_gt=None, bin_centers=None):
    """
    Computes the depth loss SEPARATELY per domain (indoor=0 / outdoor=1) and recombines
    with loss_cfg.depth_weight_indoor / depth_weight_outdoor.
    Optionally applies 1D Chamfer loss on bin_centers when using Adaptive Bins.
    """
    weights = {0: loss_cfg.depth_weight_indoor, 1: loss_cfg.depth_weight_outdoor}
    names = {0: "indoor", 1: "outdoor"}
    total = pred_depth.sum() * 0.0
    per_domain = {"indoor": None, "outdoor": None}
    any_domain = False
    for d, w in weights.items():
        dmask = (domain_gt == d)
        if dmask.sum() == 0:
            continue
        d_valid = valid_mask[dmask]
        if d == 1 and getattr(loss_cfg, "outdoor_depth_train_max", None) is not None:
            d_valid = d_valid & (gt_depth[dmask] < loss_cfg.outdoor_depth_train_max)
        # Distance-based reliability weighting: outdoor only (indoor/ScanNet's Kinect-
        # style depth doesn't have this same distance-correlated noise, and indoor's
        # whole range 0.3-10m is already close to the near_distance below). See
        # depth_reliability_weight()'s docstring for the physics + evidence behind this.
        d_weight = None
        if d == 1 and getattr(loss_cfg, "depth_reliability_weighting", True):
            d_weight = depth_reliability_weight(
                gt_depth[dmask],
                near_distance=getattr(loss_cfg, "depth_reliability_near", 15.0),
                power=getattr(loss_cfg, "depth_reliability_power", 2.0),
                min_weight=getattr(loss_cfg, "depth_reliability_min_weight", 0.15),
            )
            if seg_gt is not None and getattr(loss_cfg, "depth_class_importance_weighting", True):
                class_w = depth_class_importance_weight(
                    seg_gt[dmask],
                    important_classes=getattr(loss_cfg, "depth_important_classes", (9, 10)),
                    boost=getattr(loss_cfg, "depth_class_importance_boost", 1.5),
                    ignore_index=getattr(loss_cfg, "seg_ignore_index", 255),
                )
                d_weight = d_weight * class_w
        l_d = _single_domain_depth_loss(pred_depth[dmask], gt_depth[dmask], d_valid, loss_cfg, weight=d_weight)
        total = total + w * l_d
        per_domain[names[d]] = l_d.item()
        any_domain = True
    if not any_domain:
        # no domain labels in this batch at all (shouldn't happen, but stay safe)
        total = _single_domain_depth_loss(pred_depth, gt_depth, valid_mask, loss_cfg)

    # Adaptive bins 1D Chamfer regularisation
    chamfer_w = getattr(loss_cfg, "depth_bin_chamfer_weight", 0.0)
    if bin_centers is not None and chamfer_w > 0:
        l_chamfer = chamfer_bin_loss(bin_centers, gt_depth, valid_mask)
        total = total + chamfer_w * l_chamfer

    return total, per_domain


def multiscale_depth_loss(depth_aux_list, gt_depth, valid_mask, domain_gt, loss_cfg):
    """
    Deep supervision for models/depth_decoder.py's auxiliary heads (see
    ModelConfig.depth_aux_supervision). Each aux prediction lives at a coarser
    intermediate decoder resolution; GT/valid_mask are downsampled to match (nearest,
    to avoid inventing sub-pixel depth values) rather than upsampling the prediction.
    Weighted increasingly toward the finer (later, closer-to-final) scales -- the
    coarsest aux head is the least reliable signal, so it gets the smallest weight.
    """
    if not depth_aux_list:
        return gt_depth.sum() * 0.0
    n = len(depth_aux_list)
    # increasing weights, coarsest (first) -> finest (last), normalized to sum to 1
    raw_weights = [i + 1 for i in range(n)]
    weights = [w / sum(raw_weights) for w in raw_weights]

    total = gt_depth.sum() * 0.0
    for aux_pred, w in zip(depth_aux_list, weights):
        h, w_ = aux_pred.shape[-2:]
        gt_ds = F.interpolate(gt_depth.unsqueeze(1), size=(h, w_), mode="nearest").squeeze(1)
        valid_ds = F.interpolate(valid_mask.float().unsqueeze(1), size=(h, w_), mode="nearest").squeeze(1) > 0.5
        l_d, _ = depth_loss(aux_pred, gt_ds, valid_ds, domain_gt, loss_cfg)
        total = total + w * l_d
    return total


def edge_aware_smoothness_loss(pred_depth, image, eps=1e-3):
    """
    Standard monocular-depth edge-aware smoothness term (Godard et al. style), applied
    in LOG-depth space so it's comparable across indoor (~0-10m) and outdoor (~0-80m)
    scales. Penalizes depth gradients except where the image itself has strong
    gradients (i.e. object boundaries are allowed to have depth discontinuities;
    interiors of roads/walls/sky are pushed to be smooth).
    """
    log_depth = torch.log(pred_depth.clamp(min=eps)).unsqueeze(1)   # (B,1,H,W)

    grad_depth_x = torch.abs(log_depth[:, :, :, :-1] - log_depth[:, :, :, 1:])
    grad_depth_y = torch.abs(log_depth[:, :, :-1, :] - log_depth[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)

    grad_depth_x = grad_depth_x * torch.exp(-grad_img_x)
    grad_depth_y = grad_depth_y * torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()


# ---------------------------------------------------------------------------
# Boundary Consistency Loss: Sobel(seg boundary) vs Sobel(depth) agreement
# ---------------------------------------------------------------------------
class SobelEdge(nn.Module):
    """Computes Sobel gradient magnitude for a single-channel input."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x):
        # x: (B,1,H,W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return mag


_sobel = None


def boundary_consistency_loss(seg_logits, pred_depth, valid_mask, weight=None):
    """
    seg_logits: (B, num_classes, H, W) -- convert to a soft "boundary-ness" map
                via the entropy / max-prob edge of the softmax, then Sobel it.
    pred_depth: (B, H, W)
    valid_mask: (B, H, W) bool

    weight (optional): per-pixel reliability weight, same shape as valid_mask, e.g.
    from depth_reliability_weight(). Rationale: this loss compares depth-map edges
    against segmentation-map edges, using the GROUND-TRUTH depth's spatial location
    to decide where a "boundary" should be. At far outdoor distances, the ground-truth
    depth itself is noisy (see depth_reliability_weight's docstring for the physics),
    so a "boundary" detected there is built on unreliable supervision for exactly the
    same reason the depth loss terms are -- this is a direct, consistent extension of
    that same idea to this loss, not a new assumption. Pass None to disable and match
    prior (unweighted) behaviour exactly.
    """
    global _sobel
    if _sobel is None or _sobel.kx.device != seg_logits.device:
        _sobel = SobelEdge().to(seg_logits.device)

    seg_prob = F.softmax(seg_logits, dim=1)
    seg_conf = seg_prob.max(dim=1, keepdim=True)[0]          # (B,1,H,W) confidence map
    seg_edge = _sobel(seg_conf)

    depth_in = pred_depth.unsqueeze(1)
    depth_edge = _sobel(depth_in)

    # normalize both edge maps to [0,1] per-sample for scale-invariant comparison
    def norm(t):
        b = t.shape[0]
        flat = t.view(b, -1)
        mn = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        mx = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        return (t - mn) / (mx - mn + 1e-6)

    seg_edge_n = norm(seg_edge)
    depth_edge_n = norm(depth_edge)

    mask = valid_mask.unsqueeze(1).float()
    if weight is not None:
        mask = mask * weight.unsqueeze(1)
    diff = torch.abs(seg_edge_n - depth_edge_n) * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom


# ---------------------------------------------------------------------------
# NaN/Inf guard: if any individual loss term goes non-finite (should be rare after
# the silog_loss fix above, but this is a last-resort safety net for anything
# unforeseen), zero out just that term's contribution and print an identifying
# warning ONCE per term-name so a real recurrence is diagnosable instead of
# silently corrupting the running epoch average into "loss=nan".
# ---------------------------------------------------------------------------
_nan_warned = set()


def _guard(name, value):
    if not torch.isfinite(value):
        if name not in _nan_warned:
            # tqdm.write(), not print(): this can fire WHILE a progress bar is actively
            # rendering (mid-epoch). A bare print() there corrupts tqdm's carriage-return
            # redraw and is exactly what produces garbled/duplicated terminal output on
            # the next refresh. tqdm.write() clears the bar, writes cleanly, then redraws.
            tqdm.write(f"[WARNING] non-finite loss term '{name}' encountered (value={value.item()}); "
                       f"zeroing its contribution for this batch. This message prints once per term name; "
                       f"if you see it, please report which term(s) it was.")
            _nan_warned.add(name)
        return torch.zeros_like(value)
    return value


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------
_class_weights_cache = {}


def _load_class_weights(path, device):
    """Loads and caches per-class weights from compute_class_weights.py's output.
    Cached by (path, device) so this only reads the file once per run, not every
    single training step."""
    key = (path, str(device))
    if key not in _class_weights_cache:
        import json
        with open(path) as f:
            data = json.load(f)
        _class_weights_cache[key] = torch.tensor(data["weights"], dtype=torch.float32, device=device)
    return _class_weights_cache[key]


def total_loss(outputs, targets, loss_cfg, num_seg_classes, ignore_index):
    # Force this entire numerically-sensitive computation (log/sqrt/pow/large sums)
    # to run in real float32, REGARDLESS of an ambient torch.cuda.amp.autocast(...)
    # context. total_loss is called inside training's autocast block; without this,
    # outputs["depth"]/["seg_logits"] (produced by Conv2d/Linear layers, which DO
    # get cast to fp16 under autocast) would carry fp16's ~65504 max value into
    # this math -- e.g. a 300m-vs-5m depth error squares to 87025, which silently
    # overflows to inf in fp16 and can turn into NaN a few ops later. Forcing fp32
    # here costs a bit of extra memory/compute but eliminates that failure mode at
    # the source rather than only catching it after the fact via _guard() below.
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False):
        outputs = {k: (v.float() if torch.is_tensor(v) else [t.float() for t in v] if isinstance(v, list) else v)
                   for k, v in outputs.items()}
        targets = {k: (v.float() if torch.is_tensor(v) and torch.is_floating_point(v) else v)
                   for k, v in targets.items()}

        l_domain = _guard("domain", domain_loss(outputs["domain_logits"], targets["domain"]))
        class_weights = None
        cw_path = getattr(loss_cfg, "seg_class_weights_path", None)
        if cw_path:
            class_weights = _load_class_weights(cw_path, outputs["seg_logits"].device)
        l_seg = _guard("seg", segmentation_loss(outputs["seg_logits"], targets["seg"], num_seg_classes,
                                                 ignore_index, loss_cfg.seg_label_smoothing,
                                                 robust_weight=loss_cfg.seg_robust_weight,
                                                 robust_q=loss_cfg.seg_robust_q,
                                                 boundary_weighting=getattr(loss_cfg, "seg_boundary_weighting", True),
                                                 boundary_near=getattr(loss_cfg, "seg_boundary_near", 5),
                                                 boundary_power=getattr(loss_cfg, "seg_boundary_power", 1.5),
                                                 boundary_min_weight=getattr(loss_cfg, "seg_boundary_min_weight", 0.3),
                                                 class_weights=class_weights,
                                                 focal_weight=getattr(loss_cfg, "seg_focal_weight", 0.4),
                                                 focal_gamma=getattr(loss_cfg, "seg_focal_gamma", 2.0),
                                                 lovasz_weight=getattr(loss_cfg, "seg_lovasz_weight", 0.4)))
        l_depth, l_depth_per_domain = depth_loss(outputs["depth"], targets["depth"], targets["depth_valid"],
                                                  targets["domain"], loss_cfg, seg_gt=targets["seg"],
                                                  bin_centers=outputs.get("bin_centers"))
        l_depth = _guard("depth", l_depth)
        # guard the per-domain breakdown too -- these get stored raw via .item() below
        # and weren't previously protected even though the combined l_depth was.
        for k in ("indoor", "outdoor"):
            v = l_depth_per_domain[k]
            if v is not None and not math.isfinite(v):
                l_depth_per_domain[k] = 0.0
        l_smooth = _guard("smooth", edge_aware_smoothness_loss(outputs["depth"], targets["image"]))

        # Boundary loss reliability weighting: same distance-based weight used for the
        # depth loss terms (Section 3.5 in the methodology write-up), extended here on
        # the grounds that a "boundary" detected from noisy far-range ground-truth
        # depth is built on unreliable supervision for exactly the same reason -- a
        # direct, consistent extension of an already-evidenced idea, not a new
        # assumption. Indoor pixels always get weight 1.0 (unaffected, same as the
        # depth loss). Disabled by default via boundary_reliability_weighting=False
        # until tested as its own experiment, consistent with how every other new
        # mechanism in this project has been introduced.
        boundary_weight = None
        if getattr(loss_cfg, "boundary_reliability_weighting", False) and getattr(loss_cfg, "depth_reliability_weighting", True):
            boundary_weight = torch.ones_like(targets["depth"])
            outdoor_mask = (targets["domain"] == 1).view(-1, 1, 1)   # (B,) -> (B,1,1) to broadcast against (B,H,W)
            if outdoor_mask.any():
                boundary_weight = torch.where(
                    outdoor_mask.expand_as(boundary_weight),
                    depth_reliability_weight(
                        targets["depth"],
                        near_distance=getattr(loss_cfg, "depth_reliability_near", 15.0),
                        power=getattr(loss_cfg, "depth_reliability_power", 2.0),
                        min_weight=getattr(loss_cfg, "depth_reliability_min_weight", 0.15),
                    ),
                    boundary_weight,
                )
        l_boundary = _guard("boundary", boundary_consistency_loss(outputs["seg_logits"], outputs["depth"],
                                                                    targets["depth_valid"], weight=boundary_weight))

        total = (loss_cfg.alpha_domain * l_domain +
                 loss_cfg.beta_seg * l_seg +
                 loss_cfg.gamma_depth * l_depth +
                 loss_cfg.edge_smooth_weight * l_smooth +
                 loss_cfg.delta_boundary * l_boundary)

        loss_dict = {
            "loss_total": total.item(),
            "loss_domain": l_domain.item(),
            "loss_seg": l_seg.item(),
            "loss_depth": l_depth.item(),
            "loss_smooth": l_smooth.item(),
            "loss_boundary": l_boundary.item(),
        }

        if outputs.get("depth_aux"):
            l_depth_aux = _guard("depth_aux", multiscale_depth_loss(outputs["depth_aux"], targets["depth"],
                                                                      targets["depth_valid"], targets["domain"], loss_cfg))
            total = total + loss_cfg.depth_aux_weight * l_depth_aux
            loss_dict["loss_total"] = total.item()
            loss_dict["loss_depth_aux"] = l_depth_aux.item()

    if l_depth_per_domain["indoor"] is not None:
        loss_dict["loss_depth_indoor"] = l_depth_per_domain["indoor"]
    if l_depth_per_domain["outdoor"] is not None:
        loss_dict["loss_depth_outdoor"] = l_depth_per_domain["outdoor"]
    return total, loss_dict
