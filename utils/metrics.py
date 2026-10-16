"""
Evaluation metrics.

Segmentation: mean IoU (mIoU), per-class IoU, pixel accuracy, mean class accuracy.
Depth:        AbsRel, SqRel, RMSE, RMSE(log), log10, Delta1/Delta2/Delta3 accuracy.
Domain:       classification accuracy, per-class accuracy (indoor/outdoor).

Segmentation + domain metrics accumulate a confusion matrix over the whole
epoch/val-set (standard practice, avoids batch-size bias). Depth metrics
accumulate running sums so they are exact over the full dataset too.
"""
import numpy as np
import torch


class SegmentationMetric:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion.fill(0)

    @torch.no_grad()
    def update(self, pred_logits, target):
        """pred_logits: (B,C,H,W), target: (B,H,W)"""
        pred = pred_logits.argmax(dim=1)
        pred = pred.detach().cpu().numpy().reshape(-1)
        gt = target.detach().cpu().numpy().reshape(-1)
        valid = gt != self.ignore_index
        pred, gt = pred[valid], gt[valid]
        idx = gt * self.num_classes + pred
        cm = np.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        self.confusion += cm

    def compute(self):
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / np.maximum(union, 1e-9), np.nan)
        miou = np.nanmean(iou)

        pixel_acc = tp.sum() / max(cm.sum(), 1e-9)
        class_acc = np.where(cm.sum(1) > 0, tp / np.maximum(cm.sum(1), 1e-9), np.nan)
        mean_class_acc = np.nanmean(class_acc)

        return {
            "mIoU": float(miou),
            "mIoU_seen_classes": float(np.nanmean(iou[cm.sum(1) > 0])) if (cm.sum(1) > 0).any() else float("nan"),
            "pixel_accuracy": float(pixel_acc),
            "mean_class_accuracy": float(mean_class_acc),
            "per_class_iou": iou.tolist(),
        }


class DepthMetric:
    """Accumulates exact running sums (not batch-averaged means-of-means)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.abs_rel_sum = 0.0
        self.sq_rel_sum = 0.0
        self.rmse_sq_sum = 0.0
        self.rmse_log_sq_sum = 0.0
        self.log10_sum = 0.0
        self.d1 = 0
        self.d2 = 0
        self.d3 = 0

    @torch.no_grad()
    def update(self, pred, gt, valid_mask, eps=1e-6):
        """pred, gt, valid_mask: (B,H,W) tensors, pred/gt in meters."""
        pred = pred[valid_mask].clamp(min=eps)
        gt = gt[valid_mask].clamp(min=eps)
        if pred.numel() == 0:
            return
        n = pred.numel()

        abs_rel = torch.abs(pred - gt) / gt
        sq_rel = ((pred - gt) ** 2) / gt
        rmse_sq = (pred - gt) ** 2
        rmse_log_sq = (torch.log(pred) - torch.log(gt)) ** 2
        log10 = torch.abs(torch.log10(pred) - torch.log10(gt))

        ratio = torch.max(pred / gt, gt / pred)
        d1 = (ratio < 1.25).sum().item()
        d2 = (ratio < 1.25 ** 2).sum().item()
        d3 = (ratio < 1.25 ** 3).sum().item()

        self.n += n
        self.abs_rel_sum += abs_rel.sum().item()
        self.sq_rel_sum += sq_rel.sum().item()
        self.rmse_sq_sum += rmse_sq.sum().item()
        self.rmse_log_sq_sum += rmse_log_sq.sum().item()
        self.log10_sum += log10.sum().item()
        self.d1 += d1
        self.d2 += d2
        self.d3 += d3

    def compute(self):
        n = max(self.n, 1)
        return {
            "AbsRel": self.abs_rel_sum / n,
            "SqRel": self.sq_rel_sum / n,
            "RMSE": float(np.sqrt(self.rmse_sq_sum / n)),
            "RMSE_log": float(np.sqrt(self.rmse_log_sq_sum / n)),
            "log10": self.log10_sum / n,
            "delta1": self.d1 / n,
            "delta2": self.d2 / n,
            "delta3": self.d3 / n,
        }


class DomainMetric:
    def __init__(self, num_domains=2):
        self.num_domains = num_domains
        self.confusion = np.zeros((num_domains, num_domains), dtype=np.int64)

    def reset(self):
        self.confusion.fill(0)

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        gt = target.detach().cpu().numpy()
        idx = gt * self.num_domains + pred
        cm = np.bincount(idx, minlength=self.num_domains ** 2).reshape(self.num_domains, self.num_domains)
        self.confusion += cm

    def compute(self):
        cm = self.confusion.astype(np.float64)
        acc = np.diag(cm).sum() / max(cm.sum(), 1e-9)
        per_class_acc = np.where(cm.sum(1) > 0, np.diag(cm) / np.maximum(cm.sum(1), 1e-9), np.nan)
        return {
            "accuracy": float(acc),
            "indoor_accuracy": float(per_class_acc[0]) if len(per_class_acc) > 0 else float("nan"),
            "outdoor_accuracy": float(per_class_acc[1]) if len(per_class_acc) > 1 else float("nan"),
        }


DOMAIN_NAMES = {0: "indoor", 1: "outdoor"}


class PerDomainMetrics:
    """
    Wraps SegmentationMetric + DepthMetric so seg/depth results are tracked
    BOTH combined (all domains together) AND separately per domain
    (indoor-only, outdoor-only). This matters because:
      - indoor/outdoor depth ranges differ hugely (0.3-10m vs up to 80m+),
        so a single blended AbsRel/RMSE can hide one domain being bad.
      - indoor/outdoor segmentation classes barely overlap, so a combined
        mIoU doesn't tell you whether FiLM/cross-attention conditioning is
        actually adapting well to BOTH domains, vs. just one.
    """
    def __init__(self, num_seg_classes, ignore_index, num_domains=2):
        self.num_domains = num_domains
        self.combined_seg = SegmentationMetric(num_seg_classes, ignore_index)
        self.combined_depth = DepthMetric()
        self.per_domain_seg = {d: SegmentationMetric(num_seg_classes, ignore_index) for d in range(num_domains)}
        self.per_domain_depth = {d: DepthMetric() for d in range(num_domains)}

    def reset(self):
        self.combined_seg.reset()
        self.combined_depth.reset()
        for d in range(self.num_domains):
            self.per_domain_seg[d].reset()
            self.per_domain_depth[d].reset()

    @torch.no_grad()
    def update(self, seg_logits, seg_gt, pred_depth, depth_gt, depth_valid, domain_gt):
        self.combined_seg.update(seg_logits, seg_gt)
        self.combined_depth.update(pred_depth, depth_gt, depth_valid)

        for d in range(self.num_domains):
            mask = (domain_gt == d)
            if mask.sum() == 0:
                continue
            self.per_domain_seg[d].update(seg_logits[mask], seg_gt[mask])
            self.per_domain_depth[d].update(pred_depth[mask], depth_gt[mask], depth_valid[mask])

    def compute(self):
        result = {
            "combined": {
                "segmentation": self.combined_seg.compute(),
                "depth": self.combined_depth.compute(),
            }
        }
        for d in range(self.num_domains):
            name = DOMAIN_NAMES.get(d, f"domain_{d}")
            result[name] = {
                "segmentation": self.per_domain_seg[d].compute(),
                "depth": self.per_domain_depth[d].compute(),
            }
        return result
