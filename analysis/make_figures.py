"""Regenerate the report figures (Figures 2-4) from the raw per-run JSON files.

Usage (from the repository root):
    python analysis/make_figures.py

Writes report/figures/fig2_interaction_cosine.png, fig3_protocol_comparison.png, fig4_learning_curves.png.
Figures 5a/5b (gradient norms) come from the notebook and are copied from results/notebook_figures/.
Needs numpy and matplotlib.
"""
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stats_utils import t_crit

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

V1 = json.load(open(RESULTS / "s8_residual_depth_results.json"))
V2 = json.load(open(RESULTS / "s8_residual_depth_results_cosine.json"))
C_ON, C_OFF = "#1f77b4", "#d95f02"


def runs(R, d, s):
    return sorted((r for r in R if r["depth"] == d and r["use_residual"] == s), key=lambda r: r["seed"])


final = lambda r: r["valid_accuracies"][-1]
legacy = lambda r: r["test_acc"]

# ---- Figure 2: interaction plot (cosine LR, final epoch) ----
fig, ax = plt.subplots(figsize=(5.2, 4.0))
for s, c, lab, dx in ((True, C_ON, "Residual ON", -0.25), (False, C_OFF, "Residual OFF", 0.25)):
    means, sds = [], []
    for d in (8, 20):
        a = np.array([final(r) for r in runs(V2, d, s)]) * 100
        ax.scatter([d + dx] * len(a), a, color=c, alpha=0.45, s=22, zorder=3)
        means.append(a.mean())
        sds.append(a.std(ddof=1))
    ax.errorbar([8 + dx, 20 + dx], means, yerr=sds, color=c, marker="o", capsize=4, lw=1.6, label=lab, zorder=4)
ax.set_xticks([8, 20])
ax.set_xlabel("Depth (6n+2)")
ax.set_ylabel("Final-epoch test accuracy (%)")
ax.set_title("Cosine schedule (primary recipe), BN on, 3 seeds\nmean +/- SD (n-1); dots = individual seeds", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_interaction_cosine.png", dpi=160)
plt.close(fig)

# ---- Figure 3: effect estimate under three protocols ----
protocols = [("v1 constant LR,\nbest-test-loss epoch (selects on test set)", V1, legacy),
             ("v1 constant LR,\nfinal epoch (no selection)", V1, final),
             ("v2 cosine LR,\nfinal epoch (no selection)", V2, final)]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True, sharex=True)
for ax, d in zip(axes, (8, 20)):
    for i, (lab, R, m) in enumerate(protocols):
        a = np.array([m(r) for r in runs(R, d, True)]) * 100
        b = np.array([m(r) for r in runs(R, d, False)]) * 100
        va, vb = a.var(ddof=1) / 3, b.var(ddof=1) / 3
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2)
        tc = t_crit(df)
        diff = a.mean() - b.mean()
        col = "#2ca02c" if i == 2 else "#7f7f7f"
        ax.errorbar([diff], [i], xerr=[[tc * se], [tc * se]], fmt="o", color=col, capsize=4, lw=1.8)
        ax.text(diff, i - 0.28, f"{diff:+.2f}", ha="center", va="top", fontsize=8.5, color=col)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(range(3))
    ax.set_ylim(2.6, -0.7)
    ax.set_title(f"depth {d}", fontsize=10)
    ax.set_xlabel("Residual ON - OFF (points), 95% CI")
    ax.grid(alpha=0.25, axis="x")
axes[0].set_yticklabels([p[0] for p in protocols], fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_protocol_comparison.png", dpi=160)
plt.close(fig)

# ---- Figure 4: learning curves (cosine solid, constant-LR mean dashed) ----
fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
ep = np.arange(1, 11)
for j, d in enumerate((8, 20)):
    for s, c, lab in ((True, C_ON, "Residual ON"), (False, C_OFF, "Residual OFF")):
        r2, r1 = runs(V2, d, s), runs(V1, d, s)
        va2 = np.array([r["valid_accuracies"] for r in r2]) * 100
        tl2 = np.array([r["train_losses"] for r in r2])
        va1 = np.array([r["valid_accuracies"] for r in r1]) * 100
        tl1 = np.array([r["train_losses"] for r in r1])
        for ax, a2, a1 in ((axes[0, j], va2, va1), (axes[1, j], tl2, tl1)):
            ax.plot(ep, a2.mean(0), color=c, lw=1.9, label=lab + " (cosine)")
            ax.fill_between(ep, a2.min(0), a2.max(0), color=c, alpha=0.18)
            ax.plot(ep, a1.mean(0), color=c, lw=1.0, ls="--", alpha=0.7, label=lab + " (constant LR)")
    axes[0, j].set_title(f"depth {d}: validation accuracy (%)", fontsize=10)
    axes[1, j].set_title(f"depth {d}: training loss", fontsize=10)
    axes[1, j].set_xlabel("Epoch")
    for i in (0, 1):
        axes[i, j].grid(alpha=0.25)
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("Solid = cosine (mean; band = min-max over 3 seeds); dashed = constant-LR mean", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "fig4_learning_curves.png", dpi=160)
plt.close(fig)

# ---- Figures 5a/5b: produced by the notebook, copied here for the report ----
shutil.copy(RESULTS / "notebook_figures" / "gradient_norms_depth8.png", OUT / "fig5a_gradient_norms_depth8.png")
shutil.copy(RESULTS / "notebook_figures" / "gradient_norms_depth20.png", OUT / "fig5b_gradient_norms_depth20.png")
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
