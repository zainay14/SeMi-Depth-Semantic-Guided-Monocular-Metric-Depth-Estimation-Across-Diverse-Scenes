"""
Training entry point.

Usage:
    python train.py

All hyperparameters live in config.py. Checkpoints go to train_cfg.ckpt_dir,
TensorBoard logs + the per-epoch metrics CSV go to train_cfg.log_dir
(`tensorboard --logdir runs`). After training finishes, curve PNGs (one per
domain per metric, train vs val) are written to ./curves/ and t-SNE plots to
./tsne/ automatically -- see plot_curves.py / visualize_tsne.py to re-run
either standalone against a saved checkpoint.
"""
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import data_cfg, model_cfg, loss_cfg, train_cfg
from data.dataset import build_dataloaders
from models.full_model import MultiTaskModel
from utils.losses import total_loss
from utils.metrics import PerDomainMetrics, DomainMetric, DepthMetric
from utils.csv_logger import MetricsCSVLogger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr_lambda(warmup_epochs, total_epochs):
    def fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return fn


def depth_focus_loss_cfg(base_loss_cfg, epoch, total_epochs, focus_frac, max_mult):
    """
    Curriculum: from `focus_frac` of the way through training onward, linearly ramp
    depth_weight_outdoor up to base * max_mult by the final epoch. Segmentation is
    already at ~68-69% mIoU well before this point (see training logs) while outdoor
    depth is the metric still plateaued -- this reallocates more of the later training
    budget's gradient signal toward the domain that still needs it.
    """
    start = focus_frac * total_epochs
    if epoch < start or total_epochs <= start:
        return base_loss_cfg
    progress = (epoch - start) / max(1e-6, (total_epochs - start))
    mult = 1.0 + progress * (max_mult - 1.0)
    return replace(base_loss_cfg, depth_weight_outdoor=base_loss_cfg.depth_weight_outdoor * mult)


def _make_loss_targets(images, seg_gt, depth_gt, depth_valid, domain_gt):
    return {"domain": domain_gt, "seg": seg_gt, "depth": depth_gt,
            "depth_valid": depth_valid, "image": images}


@torch.no_grad()
def evaluate(model, loader, device, num_seg_classes, ignore_index, loss_cfg_eff=None, desc="val", use_tta=False, use_enhanced_tta=False):
    """
    use_tta=True: horizontal-flip test-time augmentation (see utils/tta.py).
    use_enhanced_tta=True: 4-pass multi-view TTA.
    Off by default (train.py's per-epoch validation calls this with use_tta=False, since
    paying 2x compute every epoch just for monitoring isn't worth it) -- turn it on
    for the FINAL number you report, e.g. `python evaluate.py --ckpt best.pt --tta`.
    """
    if loss_cfg_eff is None:
        loss_cfg_eff = loss_cfg
    model.eval()
    metrics = PerDomainMetrics(num_seg_classes, ignore_index, num_domains=model_cfg.num_domains)
    domain_metric = DomainMetric(model_cfg.num_domains)
    outdoor_capped_metric = DepthMetric()   # additional REPORTED-only metric, see LossConfig.depth_eval_cap_outdoor
    cap = loss_cfg.depth_eval_cap_outdoor
    running_loss = {}

    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = batch["image"].to(device, non_blocking=True)
        seg_gt = batch["seg"].to(device, non_blocking=True)
        depth_gt = batch["depth"].to(device, non_blocking=True)
        depth_valid = batch["depth_valid"].to(device, non_blocking=True)
        domain_gt = batch["domain"].to(device, non_blocking=True)

        if use_enhanced_tta:
            from utils.tta import forward_with_enhanced_tta
            outputs = forward_with_enhanced_tta(model, images, domain_gt=None, teacher_force_domain=False)
        elif use_tta:
            from utils.tta import forward_with_flip_tta
            outputs = forward_with_flip_tta(model, images, domain_gt=None, teacher_force_domain=False)
        else:
            outputs = model(images, domain_gt=None, teacher_force_domain=False)
        _, loss_dict = total_loss(
            outputs, _make_loss_targets(images, seg_gt, depth_gt, depth_valid, domain_gt),
            loss_cfg_eff, num_seg_classes, ignore_index,
        )
        for k, v in loss_dict.items():
            running_loss[k] = running_loss.get(k, 0.0) + v

        metrics.update(outputs["seg_logits"], seg_gt, outputs["depth"], depth_gt, depth_valid, domain_gt)
        domain_metric.update(outputs["domain_logits"], domain_gt)

        outdoor_mask = (domain_gt == 1)
        if outdoor_mask.any():
            capped_valid = depth_valid[outdoor_mask] & (depth_gt[outdoor_mask] < cap)
            outdoor_capped_metric.update(outputs["depth"][outdoor_mask], depth_gt[outdoor_mask], capped_valid)

    n = max(len(loader), 1)
    avg_loss = {k: v / n for k, v in running_loss.items()}
    results = metrics.compute()          # {"combined":{...}, "indoor":{...}, "outdoor":{...}}
    results["loss"] = avg_loss
    results["domain"] = domain_metric.compute()
    results["outdoor_capped"] = outdoor_capped_metric.compute()   # RMSE/AbsRel/etc for outdoor gt < cap meters
    return results


def print_domain_block(prefix, results):
    """Matches the console format used throughout the project's training logs:
        [train] loss=...  domain_acc=...  mIoU=...  delta1=...  delta2=...  delta3=...  RMSE=...  AbsRel=...  SqRel=...
          indoor  : mIoU=...  delta1=...  delta2=...  delta3=...  RMSE=...  RMSE_log=...  AbsRel=...  SqRel=...
          outdoor : mIoU=...  delta1=...  delta2=...  delta3=...  RMSE=...  RMSE_log=...  AbsRel=...  SqRel=...
    """
    c = results["combined"]
    print(f"  [{prefix:5s}] loss={results['loss']['loss_total']:.4f}  "
          f"domain_acc={results['domain']['accuracy']:.4f}  "
          f"mIoU={c['segmentation']['mIoU']:.4f}  delta1={c['depth']['delta1']:.4f}  "
          f"delta2={c['depth']['delta2']:.4f}  delta3={c['depth']['delta3']:.4f}  "
          f"RMSE={c['depth']['RMSE']:.4f}  AbsRel={c['depth']['AbsRel']:.4f}  SqRel={c['depth']['SqRel']:.4f}")
    for name in ("indoor", "outdoor"):
        r = results[name]
        seen_iou = r['segmentation'].get('mIoU_seen_classes', float('nan'))
        seen_str = f" [seen: {seen_iou:.4f}]" if seen_iou == seen_iou else ""
        print(f"    {name:8s}: mIoU={r['segmentation']['mIoU']:.4f}{seen_str}  "
              f"delta1={r['depth']['delta1']:.4f}  delta2={r['depth']['delta2']:.4f}  "
              f"delta3={r['depth']['delta3']:.4f}  RMSE={r['depth']['RMSE']:.4f}  "
              f"RMSE_log={r['depth']['RMSE_log']:.4f}  AbsRel={r['depth']['AbsRel']:.4f}  "
              f"SqRel={r['depth']['SqRel']:.4f}")
    if "outdoor_capped" in results and results["outdoor_capped"]["RMSE"] == results["outdoor_capped"]["RMSE"]:
        cap = loss_cfg.depth_eval_cap_outdoor
        oc = results["outdoor_capped"]
        print(f"    outdoor (gt<{cap:.0f}m only): RMSE={oc['RMSE']:.4f}  RMSE_log={oc['RMSE_log']:.4f}  "
              f"AbsRel={oc['AbsRel']:.4f}  delta1={oc['delta1']:.4f}")


def log_to_writer(writer, results, split, epoch):
    for group in ("combined", "indoor", "outdoor"):
        for task, task_metrics in results[group].items():
            for k, v in task_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{split}/{group}_{task}_{k}", v, epoch)
    for k, v in results["domain"].items():
        writer.add_scalar(f"{split}/domain_{k}", v, epoch)
    for k, v in results["loss"].items():
        writer.add_scalar(f"{split}/{k}", v, epoch)
    if "outdoor_capped" in results:
        for k, v in results["outdoor_capped"].items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"{split}/outdoor_capped_depth_{k}", v, epoch)


def composite_score(results):
    c = results["combined"]
    return (c["segmentation"]["mIoU"] + c["depth"]["delta1"]
            - c["depth"]["AbsRel"] + results["domain"]["accuracy"])


def main():
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    # Automatic persistent console log: every print()'d line (epoch summaries, warnings,
    # curriculum notices, etc.) also gets duplicated into a timestamped file under
    # train_cfg.log_dir, so a terminal resize/refresh/disconnect can never actually lose
    # anything -- you can always `less`/`grep`/`tail -f` this file regardless of what your
    # terminal is doing. Only stdout is duplicated (not stderr, where tqdm's own bar
    # rendering goes by default) so the log file stays clean text, not full of \r frames.
    os.makedirs(train_cfg.log_dir, exist_ok=True)
    _log_path = os.path.join(train_cfg.log_dir, f"console_{time.strftime('%Y%m%d_%H%M%S')}.log")
    _log_file = open(_log_path, "a")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = _Tee(sys.stdout, _log_file)
    print(f"[logging] console output also being saved to {_log_path} "
          f"(survives terminal resize/refresh/disconnect -- `tail -f {_log_path}` from anywhere)")

    train_loader, val_loader = build_dataloaders(
        data_cfg, train_cfg.batch_size, data_cfg.num_workers, data_cfg.balanced_domain_sampling)
    model = MultiTaskModel(model_cfg, data_cfg.num_seg_classes, img_size=data_cfg.img_size).to(device)

    param_groups = model.get_param_groups(train_cfg.lr, train_cfg.backbone_lr_mult, train_cfg.weight_decay, protect_seg=True)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(train_cfg.warmup_epochs, train_cfg.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=train_cfg.amp)
    writer = SummaryWriter(train_cfg.log_dir)
    csv_logger = MetricsCSVLogger(os.path.join(train_cfg.log_dir, "metrics.csv"))

    start_epoch = 0
    best_score = -1e9
    best_seg_miou = -1e9
    best_depth_delta1 = -1e9
    best_domain_acc = -1e9
    epochs_without_improvement = 0   # early stopping counter -- see train_cfg.early_stop_patience
    if train_cfg.resume and os.path.isfile(train_cfg.resume):
        ckpt = torch.load(train_cfg.resume, map_location=device, weights_only=False)
        raw_state = ckpt["model"]
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in raw_state.items() if k in model_state and v.shape == model_state[k].shape}
        architecture_changed = len(filtered_state) != len(model_state) or len(filtered_state) != len(raw_state)
        load_result = model.load_state_dict(filtered_state, strict=False)
        if architecture_changed:
            # The model gained/lost parameters since this checkpoint was saved (e.g. AdaptiveBinsHead).
            # strict=False lets existing weights (backbone, seg head, fusion blocks) load normally;
            # brand-new parameters are left at fresh init.
            print(f"[resume] model architecture changed since this checkpoint was saved:")
            if load_result.missing_keys:
                shown = load_result.missing_keys[:10]
                print(f"  {len(load_result.missing_keys)} new parameter(s) not in checkpoint, "
                      f"freshly initialized: {shown}{' ...' if len(load_result.missing_keys) > 10 else ''}")
            skipped = [k for k in raw_state if k not in filtered_state]
            if skipped:
                shown = skipped[:10]
                print(f"  {len(skipped)} checkpoint parameter(s) with mismatched shape/no longer used, "
                      f"ignored: {shown}{' ...' if len(skipped) > 10 else ''}")
            print(f"  [fine-tune] Starting fresh optimizer & warmup/cosine schedule for new depth bins head.")
            start_epoch = 0
            best_score = -1e9
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            for group in optimizer.param_groups:
                group["weight_decay"] = train_cfg.weight_decay
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_score = ckpt.get("best_score", best_score)
        print(f"Resumed weights from {train_cfg.resume} (starting at epoch {start_epoch})")

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()

        effective_loss_cfg = depth_focus_loss_cfg(
            loss_cfg, epoch, train_cfg.epochs, train_cfg.depth_focus_epoch_frac, train_cfg.depth_focus_max_mult)
        if effective_loss_cfg.depth_weight_outdoor != loss_cfg.depth_weight_outdoor:
            print(f"[depth-focus curriculum] epoch {epoch}: "
                  f"depth_weight_outdoor {loss_cfg.depth_weight_outdoor:.3f} -> "
                  f"{effective_loss_cfg.depth_weight_outdoor:.3f}")

        # accumulate TRAIN metrics over the epoch (same per-domain seg+depth metrics as
        # val) so indoor/outdoor train curves can be plotted, not just the loss scalar.
        train_metrics = PerDomainMetrics(data_cfg.num_seg_classes, data_cfg.ignore_index,
                                          num_domains=model_cfg.num_domains)
        train_domain_metric = DomainMetric(model_cfg.num_domains)
        running_loss = {}

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            seg_gt = batch["seg"].to(device, non_blocking=True)
            depth_gt = batch["depth"].to(device, non_blocking=True)
            depth_valid = batch["depth_valid"].to(device, non_blocking=True)
            domain_gt = batch["domain"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=train_cfg.amp):
                outputs = model(images, domain_gt=domain_gt, teacher_force_domain=True)
                loss, loss_dict = total_loss(
                    outputs, _make_loss_targets(images, seg_gt, depth_gt, depth_valid, domain_gt),
                    effective_loss_cfg, data_cfg.num_seg_classes, data_cfg.ignore_index,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                train_metrics.update(outputs["seg_logits"], seg_gt, outputs["depth"],
                                      depth_gt, depth_valid, domain_gt)
                train_domain_metric.update(outputs["domain_logits"], domain_gt)
            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0.0) + v

            if global_step % train_cfg.log_every == 0:
                # Only show the headline total + the four main task losses in the live bar --
                # the full breakdown (including per-domain depth, aux, etc.) still goes to
                # TensorBoard/CSV every log_every steps. A postfix with 8-9 keys routinely
                # exceeds normal terminal widths and forces tqdm's \r-based line to wrap,
                # which is what actually produces garbled/duplicated output on redraw --
                # \r only returns to the start of the CURRENT wrapped line, not the true start.
                _postfix_keys = ("loss_total", "loss_domain", "loss_seg", "loss_depth")
                pbar.set_postfix({k: f"{loss_dict[k]:.4f}" for k in _postfix_keys if k in loss_dict})
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train_step/{k}", v, global_step)
                writer.add_scalar("train/lr_head", optimizer.param_groups[1]["lr"], global_step)
            global_step += 1

        scheduler.step()

        n_batches = max(len(train_loader), 1)
        train_results = train_metrics.compute()
        train_results["loss"] = {k: v / n_batches for k, v in running_loss.items()}
        train_results["domain"] = train_domain_metric.compute()

        val_results = evaluate(model, val_loader, device, data_cfg.num_seg_classes, data_cfg.ignore_index,
                                loss_cfg_eff=effective_loss_cfg)

        print(f"\n[epoch {epoch}]")
        print_domain_block("train", train_results)
        print_domain_block("val", val_results)

        log_to_writer(writer, train_results, "train_epoch", epoch)
        log_to_writer(writer, val_results, "val", epoch)
        csv_logger.log_from_results(epoch, "train", train_results)
        csv_logger.log_from_results(epoch, "val", val_results)
        csv_logger.log(epoch, "val", "outdoor_capped", {}, val_results["outdoor_capped"])

        # ---- checkpointing: one combined-score "best.pt" (as before) PLUS separate
        # best_seg.pt / best_depth.pt / best_domain.pt tracking each task's own best
        # VAL metric independently, since the combined score can improve on one task
        # while another quietly regresses.
        #
        # last.pt / best.pt keep the FULL state (model + optimizer + scheduler) since
        # those are the ones you'd actually resume training from. best_seg.pt /
        # best_depth.pt / best_domain.pt are model-weights-only snapshots -- you'd only
        # ever load these for final evaluation/inference on that specific metric, never
        # to resume training, so there's no reason to pay ~3x the disk cost (AdamW's
        # optimizer state, momentum + variance buffers, is roughly 2x the model's own
        # size) for state you'll never use. This alone roughly triples how many epochs'
        # worth of checkpoints fit before you hit "no space left on device" again.
        c = val_results["combined"]
        score = composite_score(val_results)
        full_ckpt = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": best_score,
        }
        weights_only_ckpt = {"model": model.state_dict(), "epoch": epoch}
        torch.save(full_ckpt, os.path.join(train_cfg.ckpt_dir, "last.pt"))

        if score > best_score:
            best_score = score
            full_ckpt["best_score"] = best_score
            torch.save(full_ckpt, os.path.join(train_cfg.ckpt_dir, "best.pt"))
            print(f"  -> new best COMBINED score={score:.4f}, saved best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if c["segmentation"]["mIoU"] > best_seg_miou:
            best_seg_miou = c["segmentation"]["mIoU"]
            torch.save(weights_only_ckpt, os.path.join(train_cfg.ckpt_dir, "best_seg.pt"))
            print(f"  -> new best SEGMENTATION mIoU={best_seg_miou:.4f}, saved best_seg.pt")
        if c["depth"]["delta1"] > best_depth_delta1:
            best_depth_delta1 = c["depth"]["delta1"]
            torch.save(weights_only_ckpt, os.path.join(train_cfg.ckpt_dir, "best_depth.pt"))
            print(f"  -> new best DEPTH delta1={best_depth_delta1:.4f}, saved best_depth.pt")
        if val_results["domain"]["accuracy"] > best_domain_acc:
            best_domain_acc = val_results["domain"]["accuracy"]
            torch.save(weights_only_ckpt, os.path.join(train_cfg.ckpt_dir, "best_domain.pt"))
            print(f"  -> new best DOMAIN accuracy={best_domain_acc:.4f}, saved best_domain.pt")

        # Periodic snapshots for Stochastic Weight Averaging (Izmailov et al. 2018):
        # averaging weights from several late-training epochs is a standard, legitimate
        # technique that often gives a free 1-3% metric improvement with zero extra
        # training -- see swa_average.py (run once training is far enough along) for
        # how to actually build the averaged checkpoint from these.
        if epoch >= train_cfg.swa_start_epoch and epoch % train_cfg.swa_snapshot_every == 0:
            swa_dir = os.path.join(train_cfg.ckpt_dir, "swa_snapshots")
            os.makedirs(swa_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(swa_dir, f"epoch_{epoch}.pt"))
            print(f"  -> saved SWA snapshot for epoch {epoch}")

        # Early stopping: if train_cfg.early_stop_patience is set (not None), stop once
        # the combined VAL score hasn't improved for that many epochs in a row. This is
        # deliberately based on the SAME composite_score() that drives best.pt, not any
        # single noisy metric -- a plateau/oscillation in one metric (e.g. outdoor RMSE
        # bouncing epoch to epoch, which you've seen plenty of) won't trigger this on its
        # own unless the overall combined score also fails to improve for the full window.
        # Every best_*.pt checkpoint is already independently preserved regardless of
        # whether/when this stops the loop, so stopping early costs you nothing you'd
        # otherwise keep -- it only saves wall-clock time once further improvement has
        # genuinely stalled.
        if train_cfg.early_stop_patience is not None and epochs_without_improvement >= train_cfg.early_stop_patience:
            print(f"\n[early stopping] no improvement in combined score for "
                  f"{epochs_without_improvement} epochs (patience={train_cfg.early_stop_patience}) "
                  f"-- stopping at epoch {epoch}. best.pt / best_seg.pt / best_depth.pt / "
                  f"best_domain.pt all reflect each metric's true best regardless of this.")
            break

    writer.close()


    # ---- post-training artifacts: curve PNGs (per domain, train vs val, per metric)
    # and t-SNE plots (domain separation + within-domain class clustering).
    print("\nTraining complete. Generating curve plots and t-SNE visualizations...")
    try:
        from plot_curves import generate_curve_plots
        saved_curves = generate_curve_plots(os.path.join(train_cfg.log_dir, "metrics.csv"), "curves")
        print(f"  saved {len(saved_curves)} curve PNGs to curves/")
    except Exception as e:
        print(f"  [warn] curve plotting failed: {e}")

    try:
        from visualize_tsne import run_tsne_visualization
        # reuse the best.pt weights (falls back to current in-memory weights if best.pt
        # somehow wasn't written, e.g. a 1-epoch smoke test)
        best_path = os.path.join(train_cfg.ckpt_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
        run_tsne_visualization(model=model, split="val", max_per_domain=1000, out_dir="tsne", device=device)
    except Exception as e:
        print(f"  [warn] t-SNE visualization failed: {e}")

    # ---- final, paper-reportable numbers: best.pt + horizontal-flip TTA. This is the
    # number you'd actually cite -- separate from the per-epoch val numbers above (which
    # never used TTA, to keep training fast). See utils/tta.py for what TTA does/why.
    print("\nRunning final TTA evaluation on best.pt (this is your reportable number)...")
    try:
        import json
        best_path = os.path.join(train_cfg.ckpt_dir, "best.pt")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
        final_results = evaluate(model, val_loader, device, data_cfg.num_seg_classes, data_cfg.ignore_index,
                                  desc="final_tta", use_tta=True)
        print_domain_block("final(tta)", final_results)
        with open(os.path.join(train_cfg.log_dir, "final_eval_tta.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"  saved {train_cfg.log_dir}/final_eval_tta.json")
    except Exception as e:
        print(f"  [warn] final TTA evaluation failed: {e}")


if __name__ == "__main__":
    main()
