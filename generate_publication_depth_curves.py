"""
Generate publication-grade depth metric curves for both indoor and outdoor domains:
  1. RMSE log (RMSE_log)
  2. Absolute Relative error (AbsRel)
  3. Square Relative error (SqRel)
  4. Accuracy delta 1, 2, 3 (delta1, delta2, delta3)

Generates:
  - fig17_indoor_depth_curves.png (4-panel publication composite for indoor ScanNet)
  - fig18_outdoor_depth_curves.png (4-panel publication composite for outdoor Cityscapes)
  - fig19_cross_domain_depth_curves.png (4-panel direct indoor vs outdoor comparison)
  - Individual high-res standalone curves in both publication_figures/ and curves/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Styling configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

CSV_PATH = "runs/metrics-final.csv"
PUB_DIR = "publication_figures"
CURVES_DIR = "curves"
OVERLEAF_DIR = "overleaf_package/figures"

os.makedirs(PUB_DIR, exist_ok=True)
os.makedirs(CURVES_DIR, exist_ok=True)
os.makedirs(OVERLEAF_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Separate datasets
in_train = df[(df['domain'] == 'indoor') & (df['split'] == 'train')].sort_values('epoch')
in_val = df[(df['domain'] == 'indoor') & (df['split'] == 'val')].sort_values('epoch')

out_train = df[(df['domain'] == 'outdoor') & (df['split'] == 'train')].sort_values('epoch')
out_val = df[(df['domain'] == 'outdoor') & (df['split'] == 'val')].sort_values('epoch')
out_capped = df[(df['domain'] == 'outdoor_capped') & (df['split'] == 'val')].sort_values('epoch')

# Colors
C_IN_TRAIN = '#1f77b4'   # Steel blue
C_IN_VAL = '#0d47a1'     # Deep blue
C_OUT_TRAIN = '#2ca02c'  # Green
C_OUT_VAL = '#d95f02'    # Rust / Orange
C_CAPPED = '#7570b3'     # Purple
C_D1 = '#1b9e77'         # Teal
C_D2 = '#d95f02'         # Orange
C_D3 = '#7570b3'         # Purple

def smooth(series, span=5):
    return series.ewm(span=span, adjust=False).mean()

# ==============================================================================
# 1. COMPOSITE FIGURE: INDOOR DEPTH EVALUATION CURVES (4 PANELS)
# ==============================================================================
def create_indoor_composite():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.26, wspace=0.22)

    # Panel 1: AbsRel
    ax = axes[0, 0]
    ax.plot(in_train['epoch'], smooth(in_train['AbsRel'], 4), label='Train AbsRel', color=C_IN_TRAIN, lw=2.4, linestyle='-')
    ax.plot(in_val['epoch'], smooth(in_val['AbsRel'], 5), label='Val AbsRel (Smoothed)', color=C_IN_VAL, lw=2.4, linestyle='--')
    ax.plot(in_val['epoch'], in_val['AbsRel'], color=C_IN_VAL, lw=0.75, alpha=0.25, linestyle=':')
    ax.fill_between(in_train['epoch'], smooth(in_train['AbsRel'], 4), smooth(in_val['AbsRel'], 5), color=C_IN_TRAIN, alpha=0.08, label='Generalization Gap')
    best_val_absrel = in_val['AbsRel'].min()
    best_ep = in_val.loc[in_val['AbsRel'].idxmin(), 'epoch']
    ax.scatter(best_ep, best_val_absrel, color='red', s=55, zorder=5)
    ax.annotate(f"Best Val: {best_val_absrel:.4f}\n(Epoch {best_ep})",
                xy=(best_ep, best_val_absrel), xytext=(best_ep + 4, best_val_absrel + 0.07),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="red", lw=1),
                fontsize=9, fontweight='bold')
    ax.set_title("(a) Absolute Relative Error (AbsRel)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("AbsRel (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Panel 2: SqRel
    ax = axes[0, 1]
    ax.plot(in_train['epoch'], smooth(in_train['SqRel'], 4), label='Train SqRel', color=C_IN_TRAIN, lw=2.4, linestyle='-')
    ax.plot(in_val['epoch'], smooth(in_val['SqRel'], 5), label='Val SqRel (Smoothed)', color=C_IN_VAL, lw=2.4, linestyle='--')
    ax.plot(in_val['epoch'], in_val['SqRel'], color=C_IN_VAL, lw=0.75, alpha=0.25, linestyle=':')
    ax.fill_between(in_train['epoch'], smooth(in_train['SqRel'], 4), smooth(in_val['SqRel'], 5), color=C_IN_TRAIN, alpha=0.08)
    best_val_sqrel = in_val['SqRel'].min()
    best_ep_sq = in_val.loc[in_val['SqRel'].idxmin(), 'epoch']
    ax.scatter(best_ep_sq, best_val_sqrel, color='red', s=55, zorder=5)
    ax.annotate(f"Best Val: {best_val_sqrel:.4f}\n(Epoch {best_ep_sq})",
                xy=(best_ep_sq, best_val_sqrel), xytext=(best_ep_sq + 4, best_val_sqrel + 0.08),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="red", lw=1),
                fontsize=9, fontweight='bold')
    ax.set_title("(b) Square Relative Error (SqRel)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("SqRel (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Panel 3: RMSE_log
    ax = axes[1, 0]
    ax.plot(in_train['epoch'], smooth(in_train['RMSE_log'], 4), label=r'Train $\mathrm{RMSE}_{\log}$', color=C_IN_TRAIN, lw=2.4, linestyle='-')
    ax.plot(in_val['epoch'], smooth(in_val['RMSE_log'], 5), label=r'Val $\mathrm{RMSE}_{\log}$ (Smoothed)', color=C_IN_VAL, lw=2.4, linestyle='--')
    ax.plot(in_val['epoch'], in_val['RMSE_log'], color=C_IN_VAL, lw=0.75, alpha=0.25, linestyle=':')
    ax.fill_between(in_train['epoch'], smooth(in_train['RMSE_log'], 4), smooth(in_val['RMSE_log'], 5), color=C_IN_TRAIN, alpha=0.08)
    best_val_rl = in_val['RMSE_log'].min()
    best_ep_rl = in_val.loc[in_val['RMSE_log'].idxmin(), 'epoch']
    ax.scatter(best_ep_rl, best_val_rl, color='red', s=55, zorder=5)
    ax.annotate(f"Best Val: {best_val_rl:.4f}\n(Epoch {best_ep_rl})",
                xy=(best_ep_rl, best_val_rl), xytext=(best_ep_rl - 18, best_val_rl + 0.06),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="red", lw=1),
                fontsize=9, fontweight='bold')
    ax.set_title(r"(c) Logarithmic Root Mean Square Error ($\mathrm{RMSE}_{\log}$)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"$\mathrm{RMSE}_{\log}$ (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Panel 4: Delta Accuracies (delta1, delta2, delta3)
    ax = axes[1, 1]
    ax.plot(in_val['epoch'], smooth(in_val['delta1'], 4), label=r'Val $\delta_1 < 1.25$', color=C_D1, lw=2.4, linestyle='-')
    ax.plot(in_val['epoch'], smooth(in_val['delta2'], 4), label=r'Val $\delta_2 < 1.25^2$', color=C_D2, lw=2.4, linestyle='--')
    ax.plot(in_val['epoch'], smooth(in_val['delta3'], 4), label=r'Val $\delta_3 < 1.25^3$', color=C_D3, lw=2.4, linestyle='-.')
    ax.plot(in_train['epoch'], smooth(in_train['delta1'], 4), label=r'Train $\delta_1 < 1.25$', color=C_D1, lw=1.2, linestyle=':', alpha=0.6)
    
    final_d1 = in_val.iloc[-1]['delta1'] * 100
    final_d2 = in_val.iloc[-1]['delta2'] * 100
    final_d3 = in_val.iloc[-1]['delta3'] * 100
    ax.text(0.04, 0.10, f"Final ScanNet Val:\n  $\\delta_1 < 1.25$:   {final_d1:.1f}%\n  $\\delta_2 < 1.25^2$: {final_d2:.1f}%\n  $\\delta_3 < 1.25^3$: {final_d3:.1f}%",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#aaaaaa", lw=1),
            fontsize=9.5, fontweight='bold', family='monospace')
    ax.set_title(r"(d) Threshold Accuracy Progression ($\delta_1, \delta_2, \delta_3$)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Accuracy / Fraction of Pixels (higher is better)", fontsize=10, fontweight='semibold')
    ax.set_ylim(0.55, 1.01)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='lower right')

    out_file = os.path.join(PUB_DIR, "fig17_indoor_depth_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

# ==============================================================================
# 2. COMPOSITE FIGURE: OUTDOOR DEPTH EVALUATION CURVES (4 PANELS)
# ==============================================================================
def create_outdoor_composite():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.26, wspace=0.22)

    # Panel 1: AbsRel
    ax = axes[0, 0]
    ax.plot(out_train['epoch'], smooth(out_train['AbsRel'], 4), label='Train AbsRel (0-80m)', color=C_OUT_TRAIN, lw=2.2, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['AbsRel'], 5), label='Val AbsRel (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['AbsRel'], 5), label=r'Val AbsRel ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    
    best_capped_absrel = out_capped['AbsRel'].min()
    best_ep_c = out_capped.loc[out_capped['AbsRel'].idxmin(), 'epoch']
    ax.scatter(best_ep_c, best_capped_absrel, color='purple', s=45, zorder=5)
    ax.annotate(f"Best Capped: {best_capped_absrel:.4f}\n(Epoch {best_ep_c})",
                xy=(best_ep_c, best_capped_absrel), xytext=(best_ep_c + 6, best_capped_absrel + 0.045),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#f3e5f5", ec="purple", lw=1),
                fontsize=9, fontweight='bold')
    ax.set_title("(a) Absolute Relative Error (AbsRel)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("AbsRel (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='upper right')

    # Panel 2: SqRel
    ax = axes[0, 1]
    ax.plot(out_train['epoch'], smooth(out_train['SqRel'], 4), label='Train SqRel (0-80m)', color=C_OUT_TRAIN, lw=2.2, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['SqRel'], 5), label='Val SqRel (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['SqRel'], 5), label=r'Val SqRel ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    
    ax.set_title("(b) Square Relative Error (SqRel)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("SqRel (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='upper right')

    # Panel 3: RMSE_log
    ax = axes[1, 0]
    ax.plot(out_train['epoch'], smooth(out_train['RMSE_log'], 4), label=r'Train $\mathrm{RMSE}_{\log}$ (0-80m)', color=C_OUT_TRAIN, lw=2.2, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['RMSE_log'], 5), label=r'Val $\mathrm{RMSE}_{\log}$ (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['RMSE_log'], 5), label=r'Val $\mathrm{RMSE}_{\log}$ ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    
    best_cap_rl = out_capped['RMSE_log'].min()
    best_ep_rl = out_capped.loc[out_capped['RMSE_log'].idxmin(), 'epoch']
    ax.scatter(best_ep_rl, best_cap_rl, color='purple', s=45, zorder=5)
    ax.annotate(f"Best Capped: {best_cap_rl:.4f}\n(Epoch {best_ep_rl})",
                xy=(best_ep_rl, best_cap_rl), xytext=(best_ep_rl - 18, best_cap_rl + 0.045),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="#f3e5f5", ec="purple", lw=1),
                fontsize=9, fontweight='bold')
    ax.set_title(r"(c) Logarithmic Root Mean Square Error ($\mathrm{RMSE}_{\log}$)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel(r"$\mathrm{RMSE}_{\log}$ (lower is better)", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='upper right')

    # Panel 4: Delta Accuracies
    ax = axes[1, 1]
    # Capped vs Full delta1
    ax.plot(out_capped['epoch'], smooth(out_capped['delta1'], 4), label=r'Capped ($\leq 50$m) $\delta_1 < 1.25$', color=C_D1, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['delta1'], 4), label=r'Full (0-80m) $\delta_1 < 1.25$', color=C_D1, lw=1.8, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta2'], 4), label=r'Capped ($\leq 50$m) $\delta_2 < 1.25^2$', color=C_D2, lw=2.2, linestyle='-')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta3'], 4), label=r'Capped ($\leq 50$m) $\delta_3 < 1.25^3$', color=C_D3, lw=2.2, linestyle='-')
    
    final_d1_cap = out_capped.iloc[-1]['delta1'] * 100
    final_d2_cap = out_capped.iloc[-1]['delta2'] * 100
    final_d3_cap = out_capped.iloc[-1]['delta3'] * 100
    ax.text(0.04, 0.10, f"Final Cityscapes Val (Capped $\\leq 50$m):\n  $\\delta_1 < 1.25$:   {final_d1_cap:.1f}%\n  $\\delta_2 < 1.25^2$: {final_d2_cap:.1f}%\n  $\\delta_3 < 1.25^3$: {final_d3_cap:.1f}%",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#aaaaaa", lw=1),
            fontsize=9.5, fontweight='bold', family='monospace')
    ax.set_title(r"(d) Threshold Accuracy Progression ($\delta_1, \delta_2, \delta_3$)", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='bold')
    ax.set_ylabel("Accuracy / Fraction of Pixels (higher is better)", fontsize=10, fontweight='bold')
    ax.set_ylim(0.70, 1.01)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8.5, loc='lower right')

    out_file = os.path.join(PUB_DIR, "fig18_outdoor_depth_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

# ==============================================================================
# 3. COMPOSITE FIGURE: CROSS-DOMAIN DIRECT COMPARISON CURVES (4 PANELS)
# ==============================================================================
def create_cross_domain_composite():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.96, hspace=0.26, wspace=0.22)

    # Subplot (a): AbsRel Comparison
    ax = axes[0, 0]
    ax.plot(in_val['epoch'], smooth(in_val['AbsRel'], 5), label='Indoor Val AbsRel (ScanNet)', color=C_IN_VAL, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['AbsRel'], 5), label='Outdoor Val AbsRel (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['AbsRel'], 5), label=r'Outdoor Val AbsRel ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    ax.set_title("(a) Absolute Relative Error (AbsRel) Trajectories", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("AbsRel (lower is better)", fontsize=10, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Subplot (b): SqRel Comparison
    ax = axes[0, 1]
    ax.plot(in_val['epoch'], smooth(in_val['SqRel'], 5), label='Indoor Val SqRel (ScanNet)', color=C_IN_VAL, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['SqRel'], 5), label='Outdoor Val SqRel (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['SqRel'], 5), label=r'Outdoor Val SqRel ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    ax.set_title("(b) Square Relative Error (SqRel) Trajectories", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("SqRel (lower is better)", fontsize=10, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Subplot (c): RMSE_log Comparison
    ax = axes[1, 0]
    ax.plot(in_val['epoch'], smooth(in_val['RMSE_log'], 5), label=r'Indoor Val $\mathrm{RMSE}_{\log}$ (ScanNet)', color=C_IN_VAL, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['RMSE_log'], 5), label=r'Outdoor Val $\mathrm{RMSE}_{\log}$ (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['RMSE_log'], 5), label=r'Outdoor Val $\mathrm{RMSE}_{\log}$ ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    ax.set_title(r"(c) Logarithmic RMSE ($\mathrm{RMSE}_{\log}$) Trajectories", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel(r"$\mathrm{RMSE}_{\log}$ (lower is better)", fontsize=10, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='upper right')

    # Subplot (d): Threshold Accuracy delta1 Comparison
    ax = axes[1, 1]
    ax.plot(in_val['epoch'], smooth(in_val['delta1'], 4), label=r'Indoor Val $\delta_1 < 1.25$ (ScanNet)', color=C_IN_VAL, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['delta1'], 4), label=r'Outdoor Val $\delta_1 < 1.25$ (0-80m Raw)', color=C_OUT_VAL, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta1'], 4), label=r'Outdoor Val $\delta_1 < 1.25$ ($\leq 50$m Capped)', color=C_CAPPED, lw=2.4, linestyle='-.')
    ax.axhline(0.85, color='gray', linestyle=':', lw=1.2, label='85% Precision Threshold')
    ax.set_title(r"(d) Primary Threshold Accuracy ($\delta_1 < 1.25$) Trajectories", fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel("Epoch", fontsize=10, fontweight='semibold')
    ax.set_ylabel("Fraction of Pixels with Ratio < 1.25", fontsize=10, fontweight='semibold')
    ax.set_ylim(0.60, 0.95)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right')

    out_file = os.path.join(PUB_DIR, "fig19_cross_domain_depth_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

# ==============================================================================
# 4. INDIVIDUAL HIGH-RESOLUTION CURVES (STANDALONE)
# ==============================================================================
def create_individual_curves():
    # Helper to plot single curve
    def plot_single(x_train, y_train, x_val, y_val, title, ylabel, filename,
                    y_capped=None, label_train='Train', label_val='Val', label_capped=None,
                    c_train='#1f77b4', c_val='#0d47a1', c_capped='#7570b3', ylim=None):
        fig, ax = plt.subplots(figsize=(8, 5.2))
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.95)
        
        if y_train is not None and len(y_train) > 0:
            ax.plot(x_train, smooth(y_train, 4), label=label_train, color=c_train, lw=2.2, linestyle='-')
        if y_val is not None and len(y_val) > 0:
            ax.plot(x_val, smooth(y_val, 5), label=label_val, color=c_val, lw=2.2, linestyle='--')
        if y_capped is not None and len(y_capped) > 0:
            ax.plot(x_val, smooth(y_capped, 5), label=label_capped, color=c_capped, lw=2.4, linestyle='-.')
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Training Epoch", fontsize=11, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold')
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=10, loc='best')
        
        for d in [PUB_DIR, CURVES_DIR, OVERLEAF_DIR]:
            fig.savefig(os.path.join(d, filename), dpi=300)
        plt.close(fig)
        print(f"Saved standalone: {filename}")

    # --- Indoor Individual Curves ---
    plot_single(in_train['epoch'], in_train['RMSE_log'], in_val['epoch'], in_val['RMSE_log'],
                r"Indoor Depth $\mathrm{RMSE}_{\log}$ Convergence Curve (ScanNet)",
                r"$\mathrm{RMSE}_{\log}$ (log error)", "indoor_depth_rmse_log.png",
                c_train=C_IN_TRAIN, c_val=C_IN_VAL)

    plot_single(in_train['epoch'], in_train['AbsRel'], in_val['epoch'], in_val['AbsRel'],
                "Indoor Depth Absolute Relative Error (AbsRel) Curve (ScanNet)",
                "AbsRel Error", "indoor_depth_absrel.png",
                c_train=C_IN_TRAIN, c_val=C_IN_VAL)

    plot_single(in_train['epoch'], in_train['SqRel'], in_val['epoch'], in_val['SqRel'],
                "Indoor Depth Square Relative Error (SqRel) Curve (ScanNet)",
                "SqRel Error", "indoor_depth_sqrel.png",
                c_train=C_IN_TRAIN, c_val=C_IN_VAL)

    # Indoor delta1, delta2, delta3 standalone
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.95)
    ax.plot(in_val['epoch'], smooth(in_val['delta1'], 4), label=r'Val $\delta_1 < 1.25$', color=C_D1, lw=2.4, linestyle='-')
    ax.plot(in_val['epoch'], smooth(in_val['delta2'], 4), label=r'Val $\delta_2 < 1.25^2$', color=C_D2, lw=2.4, linestyle='--')
    ax.plot(in_val['epoch'], smooth(in_val['delta3'], 4), label=r'Val $\delta_3 < 1.25^3$', color=C_D3, lw=2.4, linestyle='-.')
    ax.plot(in_train['epoch'], smooth(in_train['delta1'], 4), label=r'Train $\delta_1 < 1.25$', color=C_D1, lw=1.2, linestyle=':', alpha=0.6)
    ax.set_title(r"Indoor Depth Accuracy Threshold Curves ($\delta_1, \delta_2, \delta_3$ on ScanNet)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Training Epoch", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Fraction of Pixels", fontsize=11, fontweight='semibold')
    ax.set_ylim(0.55, 1.01)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='lower right')
    for d in [PUB_DIR, CURVES_DIR, OVERLEAF_DIR]:
        fig.savefig(os.path.join(d, "indoor_depth_delta.png"), dpi=300)
    plt.close(fig)
    print("Saved standalone: indoor_depth_delta.png")

    # --- Outdoor Individual Curves ---
    plot_single(out_train['epoch'], out_train['RMSE_log'], out_val['epoch'], out_val['RMSE_log'],
                r"Outdoor Depth Logarithmic Error ($\mathrm{RMSE}_{\log}$) Curve (Cityscapes)",
                r"$\mathrm{RMSE}_{\log}$ (log error)", "outdoor_depth_rmse_log.png",
                y_capped=out_capped['RMSE_log'],
                label_train='Train (0-80m)', label_val='Val (0-80m Raw)', label_capped=r'Val ($\leq 50$m Capped)',
                c_train=C_OUT_TRAIN, c_val=C_OUT_VAL, c_capped=C_CAPPED)

    plot_single(out_train['epoch'], out_train['AbsRel'], out_val['epoch'], out_val['AbsRel'],
                "Outdoor Depth Absolute Relative Error (AbsRel) Curve (Cityscapes)",
                "AbsRel Error", "outdoor_depth_absrel.png",
                y_capped=out_capped['AbsRel'],
                label_train='Train (0-80m)', label_val='Val (0-80m Raw)', label_capped=r'Val ($\leq 50$m Capped)',
                c_train=C_OUT_TRAIN, c_val=C_OUT_VAL, c_capped=C_CAPPED)

    plot_single(out_train['epoch'], out_train['SqRel'], out_val['epoch'], out_val['SqRel'],
                "Outdoor Depth Square Relative Error (SqRel) Curve (Cityscapes)",
                "SqRel Error", "outdoor_depth_sqrel.png",
                y_capped=out_capped['SqRel'],
                label_train='Train (0-80m)', label_val='Val (0-80m Raw)', label_capped=r'Val ($\leq 50$m Capped)',
                c_train=C_OUT_TRAIN, c_val=C_OUT_VAL, c_capped=C_CAPPED)

    # Outdoor delta1, delta2, delta3 standalone
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.95)
    ax.plot(out_capped['epoch'], smooth(out_capped['delta1'], 4), label=r'Capped ($\leq 50$m) $\delta_1 < 1.25$', color=C_D1, lw=2.4, linestyle='-')
    ax.plot(out_val['epoch'], smooth(out_val['delta1'], 4), label=r'Raw (0-80m) $\delta_1 < 1.25$', color=C_D1, lw=1.8, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta2'], 4), label=r'Capped ($\leq 50$m) $\delta_2 < 1.25^2$', color=C_D2, lw=2.2, linestyle='-')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta3'], 4), label=r'Capped ($\leq 50$m) $\delta_3 < 1.25^3$', color=C_D3, lw=2.2, linestyle='-')
    ax.set_title(r"Outdoor Depth Accuracy Threshold Curves ($\delta_1, \delta_2, \delta_3$ on Cityscapes)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Training Epoch", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Fraction of Pixels", fontsize=11, fontweight='semibold')
    ax.set_ylim(0.70, 1.01)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='lower right')
    for d in [PUB_DIR, CURVES_DIR, OVERLEAF_DIR]:
        fig.savefig(os.path.join(d, "outdoor_depth_delta.png"), dpi=300)
    plt.close(fig)
    print("Saved standalone: outdoor_depth_delta.png")

    # --- Outdoor Capped Dedicated Standalone Curves ---
    plot_single(None, None, out_capped['epoch'], out_capped['RMSE_log'],
                r"Outdoor Capped ($\leq 50$m) Depth $\mathrm{RMSE}_{\log}$ Curve (Cityscapes)",
                r"$\mathrm{RMSE}_{\log}$ (log error)", "outdoor_capped_depth_rmse_log.png",
                label_val=r'Val ($\leq 50$m Capped)', c_val=C_CAPPED)

    plot_single(None, None, out_capped['epoch'], out_capped['AbsRel'],
                r"Outdoor Capped ($\leq 50$m) Depth Absolute Relative Error (AbsRel) Curve",
                "AbsRel Error", "outdoor_capped_depth_absrel.png",
                label_val=r'Val ($\leq 50$m Capped)', c_val=C_CAPPED)

    plot_single(None, None, out_capped['epoch'], out_capped['SqRel'],
                r"Outdoor Capped ($\leq 50$m) Depth Square Relative Error (SqRel) Curve",
                "SqRel Error", "outdoor_capped_depth_sqrel.png",
                label_val=r'Val ($\leq 50$m Capped)', c_val=C_CAPPED)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.12, right=0.95)
    ax.plot(out_capped['epoch'], smooth(out_capped['delta1'], 4), label=r'Capped ($\leq 50$m) $\delta_1 < 1.25$', color=C_D1, lw=2.4, linestyle='-')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta2'], 4), label=r'Capped ($\leq 50$m) $\delta_2 < 1.25^2$', color=C_D2, lw=2.2, linestyle='--')
    ax.plot(out_capped['epoch'], smooth(out_capped['delta3'], 4), label=r'Capped ($\leq 50$m) $\delta_3 < 1.25^3$', color=C_D3, lw=2.2, linestyle='-.')
    ax.set_title(r"Outdoor Capped ($\leq 50$m) Depth Accuracy Threshold Curves ($\delta_1, \delta_2, \delta_3$)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Training Epoch", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Fraction of Pixels", fontsize=11, fontweight='semibold')
    ax.set_ylim(0.75, 1.01)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10, loc='lower right')
    for d in [PUB_DIR, CURVES_DIR, OVERLEAF_DIR]:
        fig.savefig(os.path.join(d, "outdoor_capped_depth_delta.png"), dpi=300)
    plt.close(fig)
    print("Saved standalone: outdoor_capped_depth_delta.png")

if __name__ == "__main__":
    create_indoor_composite()
    create_outdoor_composite()
    create_cross_domain_composite()
    create_individual_curves()
    print("All depth curves successfully generated at 300 DPI!")
