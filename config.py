"""
Central configuration for the multi-task (Domain / Segmentation / Depth) pipeline.
Edit paths + hyperparameters here. Everything else reads from this file.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    root: str = os.environ.get(
        "DATASET_ROOT",
        r"c:\Users\User\Documents\Zainab Iftikhar\processed\processed"
        if os.path.isdir(r"c:\Users\User\Documents\Zainab Iftikhar\processed\processed")
        else (
            r"c:\Users\User\Documents\Zainab Iftikhar\processed"
            if os.path.isdir(r"c:\Users\User\Documents\Zainab Iftikhar\processed")
            else "/home/hashir/Thesis/Dataset/processed"
        ),
    )
    train_split: str = "train"
    val_split: str = "val"
    img_size: int = 512                      # H == W after resize/crop
    num_seg_classes: int = 15                # BUG FIX (was 40): data/prepare_dataset.py's
                                              # UNIFIED_CLASSES has exactly 15 entries. The seg
                                              # head/loss/metrics all read this value, so it must
                                              # match classes.json exactly -- see the assertion in
                                              # data/dataset.py, which now checks this automatically.
    ignore_index: int = 255
    max_depth: float = 80.0                  # meters, clip depth to [min_depth, max_depth].
                                              # Must cover BOTH domains: indoor (ScanNet, ~10m)
                                              # and outdoor (Cityscapes, up to ~80m). Setting
                                              # this too low silently invalidates outdoor depth
                                              # pixels -- verify with per-domain eval metrics.
    min_depth: float = 0.001
    depth_scale: float = 1000.0              # raw depth png stored as millimeters -> divide by this to get meters
    num_workers: int = 4                     # 4 workers optimal on Windows
    subsample_indoor_stride: int = 5         # subsample redundant indoor video frames by stride 5 (train only!)
    balanced_domain_sampling: bool = True    # WeightedRandomSampler over indoor/outdoor so
                                              # unequal NYU/Cityscapes counts don't bias the domain head
    max_domain_oversample_ratio: float = 4.0  # was 6.0. CAP on how many times more often a minority-domain
                                              # image gets sampled vs a majority-domain image, per epoch.
                                              # Full balancing (uncapped) would revisit each outdoor image
                                              # ~13.5x/epoch to match indoor's count -- real repetition of
                                              # the same 2975 images, and a real contributor to the
                                              # outdoor train/val RMSE gap (train improving steadily while
                                              # val plateaued -- classic overfitting-via-repetition
                                              # signature). 6.0 trades some domain balance for less
                                              # memorization pressure. Set to None to disable the cap and
                                              # go back to full balancing.
    strong_aug_minority_domain: bool = True  # the balanced sampler revisits the minority domain
                                              # (outdoor, 2975 imgs) more often per epoch than it would
                                              # appear naturally -- that's the SAME images repeated, which
                                              # risks overfitting on outdoor specifically (see
                                              # max_domain_oversample_ratio above, which now caps how much).
                                              # When True, minority-domain samples ALSO get a stronger
                                              # augmentation pipeline (see utils/transforms.py) so each
                                              # repeat visit looks meaningfully different -- both levers
                                              # attack the same overfitting risk from different angles.
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class ModelConfig:
    backbone_name: str = "swin_tiny_patch4_window7_224"    # reverted to match the abstract's claim
                                              # (was briefly swin_small -- see README note below if
                                              # you decide to switch back later). Same stage channel
                                              # dims [96,192,384,768] and strides [4,8,16,32] either
                                              # way (verified against timm) -- full_model.py reads
                                              # channels/strides straight off the backbone module at
                                              # runtime, not from the two fields below, so swapping
                                              # backbone_name is always a one-line, safe change.
    pretrained: bool = True
    backbone_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])  # swin-tiny/small stage dims (same)
    backbone_strides: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    decode_dim: int = 256                    # SegFormer head / DPT common embedding dim (256 matches last.pt checkpoint)
    domain_hidden: int = 128
    num_domains: int = 2                     # 0 = indoor, 1 = outdoor
    cross_attn_heads: int = 4
    film_hidden: int = 128
    seg_domain_conditioning: bool = True     # FiLM-conditions the segmentation head on
                                              # indoor/outdoor domain (same mechanism the depth
                                              # branch already uses). Added because indoor/outdoor
                                              # share one 15-class output space but the SAME class
                                              # can look very different across domains (e.g. a
                                              # Cityscapes building wall vs. an indoor room wall) --
                                              # without a domain signal, one shared classifier has
                                              # to resolve that ambiguity from texture/context alone.
                                              # Targets indoor segmentation's slow convergence
                                              # specifically (val>=train throughout -- NOT an
                                              # overfitting problem, a convergence-speed one).
    depth_aux_supervision: bool = True       # multi-scale deep supervision on the depth decoder:
                                              # adds 3 small auxiliary depth heads at the decoder's
                                              # intermediate (coarser) resolutions, each supervised
                                              # against a downsampled GT. Standard technique in
                                              # monocular depth nets (e.g. BTS, LapDepth) for pushing
                                              # error down at every scale instead of only the final one.
    depth_decoder_extra_capacity: bool = True  # adds a 3rd ResidualConvUnit per decoder fusion stage
                                              # + widens the final head's mid channels 32->64. This is
                                              # a DECODER-side capacity increase, not a backbone change
                                              # -- the encoder stays exactly Swin-Tiny, so it doesn't
                                              # touch what you've already told reviewers. ~+2-3M params,
                                              # modest compute increase.
    depth_decoder_dropout: float = 0.0       # was 0.1, then reverted to 0.0 -- see weight_decay note below
                                              # fusion blocks + final head. Regularizer specifically aimed
                                              # at the outdoor overfitting pattern (train RMSE improving
                                              # steadily, val RMSE plateauing) -- automatically inactive
                                              # at eval()/inference time, no effect there. Set to 0.0 to
                                              # disable.
    use_depth_bins: bool = True               # Adaptive Bins Discretization Head (Mini-AdaBins / ARD Module)
    num_depth_bins: int = 128                 # 128 discrete adaptive bins spanning scene depth range


@dataclass
class LossConfig:
    alpha_domain: float = 0.2
    beta_seg: float = 1.0
    gamma_depth: float = 1.0
    delta_boundary: float = 0.5
    seg_label_smoothing: float = 0.05
    seg_robust_weight: float = 0.2           # blends in Generalized Cross Entropy (Zhang & Sabuncu 2018)
                                              # alongside standard CE, so a single pixel with a wrong/noisy
                                              # label can't dominate the batch gradient the way unbounded
                                              # -log(p) can. Conservative default (mostly still standard
                                              # CE) since segmentation is already performing well -- raise
                                              # toward 0.5-1.0 if you have specific evidence of label noise,
                                              # lower toward 0.0 to fully disable and match old behavior.
    seg_robust_q: float = 0.7                # GCE's q parameter (paper's recommended default). Closer to
                                              # 0 -> more CE-like (less bounded); closer to 1 -> more
                                              # MAE-like (more bounded, more robust, slower clean-label fit)
    depth_silog_lambda: float = 0.85         # standard scale-invariant log loss lambda

    # --- Adaptive Bins & Pure Log-Space Supervision (eliminates far-range disparity noise) ---
    depth_mse_weight: float = 0.0            # 0.0 when using Adaptive Bins (pure log-space SILog eliminates quadratic outlier bias)
    depth_bin_chamfer_weight: float = 0.1    # 1D Chamfer loss between predicted bin centers and GT depths
    depth_weight_indoor: float = 1.0         # per-domain multiplier on the depth loss. Outdoor plateaus
    depth_weight_outdoor: float = 1.6        # around epoch ~20 while indoor keeps improving (see logs) --
                                              # upweighting outdoor's contribution to the depth loss gives
                                              # it relatively more gradient signal per step.
    depth_aux_weight: float = 0.30           # weight on the multi-scale deep-supervision aux losses
                                              # (see ModelConfig.depth_aux_supervision)
    depth_use_berhu: bool = True             # replace plain L1 with BerHu (reverse Huber) loss:
                                              # L1 for small errors, L2 for large ones. Laina et al. 2016
                                              # (FCRN) showed this specifically improves RMSE over L1/L2
                                              # alone, because it lets large outlier errors get a stronger
                                              # (squared) gradient while staying well-behaved on the bulk
                                              # of small errors -- exactly the RMSE-vs-AbsRel trade-off
                                              # your outdoor domain is stuck on.
    depth_eval_cap_outdoor: float = 50.0     # meters. NOT used in training -- purely an additional
                                              # REPORTED metric (see train.py console output / CSV rows
                                              # tagged "outdoor_capped"). Cityscapes stereo-disparity depth
                                              # gets rapidly noisier past ~50m (disparity quantization error
                                              # grows ~quadratically with distance), so full-range (0-80m)
                                              # RMSE is dominated by GT noise the model can't fix. Reporting
                                              # a capped-range RMSE alongside the full-range one is standard
                                              # practice in monocular depth papers -- just disclose the cap.
    outdoor_depth_train_max: Optional[float] = None  # OFF -- train on full 0-max_depth range.
                                              # Set to e.g. 60.0 ONLY as an isolated experiment to test
                                              # the far-range noise cutoff (see prior training commentary).
                                              # When active, outdoor depth pixels beyond this distance are
                                              # excluded from the training loss entirely (SILog/BerHu/MSE
                                              # all skip them). With None: full range, reliability weighting
                                              # handles far-range noise via down-weighting (not exclusion).
    edge_smooth_weight: float = 0.04         # edge-aware smoothness on log-depth, image-gradient gated.
                                              # Regularizes flat regions (road/sky/wall) which dominate
                                              # pixel count and otherwise contribute noisy high-frequency
                                              # error that inflates RMSE.
    depth_reliability_weighting: bool = True  # per-pixel outdoor depth loss weighting based on KNOWN
                                              # ground-truth distance (see utils/losses.py's
                                              # depth_reliability_weight docstring for the full physics
                                              # derivation + evidence). Replaces Charbonnier/outlier-
                                              # detection as the primary noise-handling mechanism for
                                              # depth, after two consecutive real checkpoints showed
                                              # Charbonnier NOT improving (mildly worsening) outdoor RMSE --
                                              # this domain's noise is smoothly distance-correlated, not
                                              # random severe outliers, so a distance-based (not error-
                                              # based) confidence signal is better-matched to the actual
                                              # problem. Every pixel still contributes (floored at
                                              # depth_reliability_min_weight), nothing is discarded.
    depth_reliability_near: float = 20.0     # meters. Pixels at or below this distance get full weight
                                              # (1.0); weight falls off smoothly beyond it.
    depth_reliability_power: float = 2.0     # falloff steepness. The theoretically "optimal" inverse-
                                              # variance weighting derived from stereo error propagation
                                              # is power=4 (variance grows as depth^4) -- deliberately
                                              # using the gentler power=2 here so far pixels are
                                              # de-emphasized, not almost-excluded; see docstring.
    depth_reliability_min_weight: float = 0.15  # floor: no pixel's weight ever drops below this, so every
                                              # pixel keeps contributing real (if reduced) gradient signal.
    depth_class_importance_weighting: bool = True  # multiplies depth_reliability_weight by a semantic-class
                                              # importance factor: pure distance-based weighting treats a far
                                              # pedestrian identically to a far patch of empty road, even though
                                              # getting the pedestrian's depth wrong matters more in practice.
                                              # See utils/losses.py's depth_class_importance_weight docstring.
    depth_important_classes: tuple = (9, 10)  # (person, vehicle) in this project's 15-class scheme -- the
                                              # classes that get the trust boost below, regardless of distance.
    depth_class_importance_boost: float = 1.5  # important-class pixels get this multiplier on top of their
                                              # distance-based weight (e.g. a far pedestrian keeps 1.5x the
                                              # trust of a same-distance road pixel).
    seg_boundary_weighting: bool = True      # weights the segmentation cross-entropy/GCE terms by distance
                                              # from the nearest ground-truth label boundary -- pixels right at
                                              # a class transition are inherently less certain (annotation
                                              # ambiguity) than pixels deep inside a confidently-labelled
                                              # region. Computed from the GT label geometry only (not learned,
                                              # not from model predictions) -- see utils/losses.py's
                                              # seg_boundary_distance_weight docstring for the full reasoning.
    seg_boundary_near: int = 5                # pixels this many (discrete, chessboard-distance) steps or more
                                              # from the nearest boundary get full trust (1.0).
    seg_boundary_power: float = 1.5           # falloff steepness approaching the boundary.
    seg_boundary_min_weight: float = 0.3      # floor: boundary pixels themselves are never fully excluded,
                                              # just down-weighted -- they still carry real information.
    boundary_reliability_weighting: bool = False  # extends the SAME distance-based depth reliability weight
                                              # (depth_reliability_weighting above) to the boundary consistency
                                              # loss too -- a boundary detected from noisy far-range depth is
                                              # unreliable supervision for the same reason the depth loss terms
                                              # are. Defaults OFF: this is a new, untested extension, to be
                                              # enabled deliberately as its own experiment, not silently bundled
                                              # into whatever else is currently running. Has no effect unless
                                              # depth_reliability_weighting is also True.
    seg_class_weights_path: Optional[str] = "class_weights.json"  # path to compute_class_weights.py's output JSON. Addresses
                                              # a real, diagnosed problem: this project's own eval showed
                                              # several classes (ceiling, indoor person, outdoor window) at
                                              # exactly 0.0 IoU -- a class-imbalance signature, not a boundary
                                              # or confidence problem, so boundary_weighting/GCE don't address
                                              # it. None = disabled (matches prior behaviour exactly). Set to
                                              # e.g. "class_weights.json" after running compute_class_weights.py
                                              # against your real training data -- test as its OWN experiment,
                                              # same discipline as every other mechanism in this project.

    # --- Focal Loss + Lovász-Softmax (mIoU improvements) ---
    seg_focal_weight: float = 0.4            # Focal Loss (Lin et al., CVPR 2017): blends into CE+Dice.
                                              # Down-weights well-classified easy pixels so rare/hard classes
                                              # (ceiling=0.0 IoU, person=0.0 IoU) get proportionally MORE
                                              # gradient. 0.0 = disabled (pure CE+Dice as before).
    seg_focal_gamma: float = 2.0             # Focal exponent. 2.0 = paper default. Higher = stronger focus
                                              # on hard examples (try 1.5-3.0).
    seg_lovasz_weight: float = 0.4           # Lovász-Softmax (Berman et al., CVPR 2018): directly minimises
                                              # 1-mIoU, the exact metric we report. CE/Dice both optimise
                                              # proxy objectives -- Lovász is the gold-standard direct fix
                                              # for low mIoU. 0.0 = disabled.


@dataclass
class TrainConfig:
    epochs: int = 15                          # 15 epochs for adaptive bins fine-tuning
    batch_size: int = 8
    lr: float = 1e-4                          # base LR for new AdaptiveBinsHead
    backbone_lr_mult: float = 0.1            # backbone & seg head train at 1e-5 to preserve 65.91% mIoU
    weight_decay: float = 0.02
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 20
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./runs"
    resume: str = "./checkpoints/best.pt"    # fine-tune directly from our best checkpoint
    seed: int = 42
    device: str = "cuda"
    depth_focus_epoch_frac: float = 0.0      # focus on depth from epoch 0
                                              # depth_weight_outdoor UP further (curriculum: seg is
                                              # already at ~68-69% mIoU well before this point in your
                                              # logs, outdoor depth is the thing still plateaued -- spend
                                              # more of the remaining training budget where it's needed).
                                              # Starts earlier than before (was 0.5) since with the
                                              # backbone now fixed at Swin-Tiny, curriculum + decoder
                                              # capacity are the main remaining levers for outdoor RMSE.
    depth_focus_max_mult: float = 1.3        # was 2.0 -- capped lower after 12 epochs of real data showed
                                              # by up to this much on top of its base value (was 1.6)
    swa_start_epoch: int = 50                # start saving periodic full-weight snapshots this late in
                                              # training (after the depth-focus curriculum has kicked in
                                              # and the model is well into fine-tuning) for optional
                                              # Stochastic Weight Averaging -- see swa_average.py
    swa_snapshot_every: int = 2               # save a snapshot every N epochs from swa_start_epoch onward
    early_stop_patience: Optional[int] = 12  # None = disabled (default, runs the full `epochs`
                                              # schedule regardless). If set to an integer, stops
                                              # once the combined val score hasn't improved for that
                                              # many epochs in a row. Every best_*.pt checkpoint is
                                              # saved independently of this either way -- this only
                                              # affects wall-clock time, never which weights you end
                                              # up with. Set generously (e.g. 8-10+) given your
                                              # metrics have shown real epoch-to-epoch noise -- a
                                              # short patience risks stopping during normal
                                              # oscillation, not a genuine plateau.


data_cfg = DataConfig()
model_cfg = ModelConfig()
loss_cfg = LossConfig()
train_cfg = TrainConfig()

os.makedirs(train_cfg.ckpt_dir, exist_ok=True)
os.makedirs(train_cfg.log_dir, exist_ok=True)
