"""
Flat CSV logger for per-epoch metrics, split by (split=train/val) x (domain=indoor/outdoor).

One row per (epoch, split, domain) written incrementally during training, e.g.:

    epoch,split,domain,mIoU,pixel_accuracy,mean_class_accuracy,AbsRel,SqRel,RMSE,RMSE_log,log10,delta1,delta2,delta3
    0,train,indoor,0.0426,...
    0,train,outdoor,0.1462,...
    0,val,indoor,0.1491,...
    0,val,outdoor,0.4489,...

This single file is the source of truth that plot_curves.py reads to produce the
separate indoor/outdoor, train/val curve PNGs requested in the README.
"""
import csv
import os

METRIC_COLUMNS = [
    "mIoU", "pixel_accuracy", "mean_class_accuracy",
    "AbsRel", "SqRel", "RMSE", "RMSE_log", "log10", "delta1", "delta2", "delta3",
]
FIELDNAMES = ["epoch", "split", "domain"] + METRIC_COLUMNS


class MetricsCSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.isfile(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log(self, epoch, split, domain, seg_metrics, depth_metrics):
        row = {"epoch": epoch, "split": split, "domain": domain}
        row.update({k: seg_metrics.get(k, "") for k in ("mIoU", "pixel_accuracy", "mean_class_accuracy")})
        row.update({k: depth_metrics.get(k, "") for k in
                     ("AbsRel", "SqRel", "RMSE", "RMSE_log", "log10", "delta1", "delta2", "delta3")})
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    def log_from_results(self, epoch, split, results):
        """results: one of val_results / train_results from train.py, i.e. the dict
        returned by utils.metrics.PerDomainMetrics.compute() (has 'indoor'/'outdoor' keys)."""
        for domain in ("indoor", "outdoor"):
            self.log(epoch, split, domain,
                      results[domain]["segmentation"], results[domain]["depth"])
