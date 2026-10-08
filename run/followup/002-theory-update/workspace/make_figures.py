#!/usr/bin/env python3
"""Figures for 002-theory-update. Reads results/fit_results.json (from 001 artifacts)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/002-theory-update/results"
R = json.load(open(os.path.join(OUT, "fit_results.json")))

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 150,
    "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})
P = R["params_M"]
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
TIER2 = ["Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-1.7B"]
short = {"gpt2": "s", "gpt2-medium": "m", "gpt2-large": "l", "gpt2-xl": "xl",
         "EleutherAI/pythia-70m": "70m", "EleutherAI/pythia-410m": "410m",
         "EleutherAI/pythia-1.4b": "1.4b", "EleutherAI/pythia-2.8b": "2.8b",
         "Qwen/Qwen2.5-0.5B": "Qwen0.5B", "HuggingFaceTB/SmolLM2-1.7B": "SmolLM1.7B"}

faith = R["series"]["faithfulness_FC_over_FM"]
FMpos = R["series"]["faithfulness_interpretable_FM_gt0"]
ld = R["series"]["baseline_logit_diff"]

# ---------------- Figure 1: faithfulness vs scale ----------------
fig, ax = plt.subplots()
# GPT-2 F_M>0 points (fill except xl which is 3-class partial)
gp = [m for m in GPT2 if FMpos[m]]
xg = [np.log10(P[m]) for m in gp]; yg = [faith[m] for m in gp]
for m in gp:
    partial = (m == "gpt2-xl")
    ax.scatter(np.log10(P[m]), faith[m], s=90, color=OKABE_ITO[0],
               facecolors="none" if partial else OKABE_ITO[0], zorder=5,
               edgecolors=OKABE_ITO[0], linewidths=1.8)
    ax.annotate(short[m], (np.log10(P[m]), faith[m]), textcoords="offset points",
                xytext=(6, 6), fontsize=10, color=OKABE_ITO[0])
# GPT-2 fit line (n=4)
f = R["fits"]["faithfulness_vs_logparams_gpt2_FMpos"]
xs = np.linspace(min(xg) - 0.1, max(xg) + 0.1, 50)
ax.plot(xs, f["slope_per_decade"] * xs + f["intercept"], color=OKABE_ITO[0], ls="-",
        lw=1.6, alpha=0.8,
        label=f"GPT-2 fit: slope={f['slope_per_decade']:.2f}/decade, R²={f['r2']:.2f} (n=4, p={f['p_value']:.2f})")
# Pythia-1.4b (only F_M>0 Pythia)
pm = "EleutherAI/pythia-1.4b"
ax.scatter(np.log10(P[pm]), faith[pm], s=110, marker="D", color=OKABE_ITO[1], zorder=6,
           label="Pythia-1.4b (only Pythia with F(M)>0)")
ax.annotate(short[pm], (np.log10(P[pm]), faith[pm]), textcoords="offset points",
            xytext=(6, -14), fontsize=10, color=OKABE_ITO[1])
# anchor line
ax.axhline(0.878, color="0.4", ls="--", lw=1.2, label="GPT-2-small anchor 87.8% (step8, on-disk)")
ax.set_xlabel("log₁₀(parameters, millions)")
ax.set_ylabel("Discovered-circuit faithfulness  F(C)/F(M)")
ax.set_title("Faithfulness of 001's discovered circuit vs model scale")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax.text(0.01, 0.02, "Pythia 70m/410m/2.8b omitted: F(M)≤0 (model fails IOI) → ratio undefined",
        transform=ax.transAxes, fontsize=8, color="0.35")
fig.savefig(os.path.join(OUT, "fig1_faithfulness_vs_scale.png"), facecolor="white")
plt.close(fig)

# ---------------- Figure 2: behavioral logit-diff vs scale ----------------
fig, ax = plt.subplots()
def plot_fam(models, color, marker, label):
    xs = [np.log10(P[m]) for m in models]; ys = [ld[m] for m in models]
    ax.plot(xs, ys, marker=marker, color=color, ms=9, lw=1.6, label=label)
    for m in models:
        ax.annotate(short[m], (np.log10(P[m]), ld[m]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9, color=color)
plot_fam(GPT2, OKABE_ITO[0], "o", "GPT-2 family (learned pos-emb, 2019)")
plot_fam(PYTHIA, OKABE_ITO[3], "s", "Pythia family (rotary, 2023)")
plot_fam(TIER2, OKABE_ITO[2], "^", "Modern small (Tier-2, behavioural-only)")
ax.axhline(0, color="0.4", ls="--", lw=1.0)
ax.axhline(3.49, color="0.6", ls=":", lw=1.0, label="GPT-2-small anchor 3.49 (step2, on-disk)")
ax.set_xlabel("log₁₀(parameters, millions)")
ax.set_ylabel("Baseline IOI mean logit-diff (IO − S)")
ax.set_title("IOI behaviour vs scale: generation dominates size")
ax.legend(fontsize=8.5, loc="center right", framealpha=0.9)
fig.savefig(os.path.join(OUT, "fig2_behavioral_vs_scale.png"), facecolor="white")
plt.close(fig)

# ---------------- Figure 3: class-persistence heatmap (strong heads) ----------------
CLASSES = ["name mover", "s2 inhibition", "induction",
           "duplicate token", "previous token", "negative"]
COLS = GPT2 + PYTHIA
sm = R["persistence_matrix_strong_heads"]
M = np.full((len(CLASSES), len(COLS)), np.nan)
for i, c in enumerate(CLASSES):
    for j, m in enumerate(COLS):
        v = sm[m][c]
        if v is not None:
            M[i, j] = v
fig, ax = plt.subplots(figsize=(8.5, 4.6))
cmap = LinearSegmentedColormap.from_list("wg", ["#f7f7f7", "#0072B2"])
cmap.set_bad("#d9d9d9")
im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(M))
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([short[m] for m in COLS])
ax.set_yticks(range(len(CLASSES)))
ax.set_yticklabels(CLASSES)
# family group labels (just above the top row, below the title)
ax.text(1.5, -0.72, "GPT-2 (2019)", ha="center", fontsize=10, color=OKABE_ITO[0], fontweight="bold")
ax.text(5.5, -0.72, "Pythia (2023)", ha="center", fontsize=10, color=OKABE_ITO[3], fontweight="bold")
ax.axvline(3.5, color="white", lw=3)
for i in range(len(CLASSES)):
    for j in range(len(COLS)):
        if np.isnan(M[i, j]):
            ax.text(j, i, "n/r", ha="center", va="center", fontsize=8, color="0.45")
        else:
            v = int(M[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=10,
                    color="white" if v >= np.nanmax(M) * 0.55 else "0.15")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("# strong heads (|Δlogit-diff|≥0.2)")
ax.set_title("Class-persistence: strong IOI heads per class × model\n"
             "(n/r = config not run in 001; grey)", fontsize=11, pad=26)
fig.subplots_adjust(top=0.80)
fig.savefig(os.path.join(OUT, "fig3_class_persistence_heatmap.png"), facecolor="white")
plt.close(fig)

print("Wrote fig1_faithfulness_vs_scale.png, fig2_behavioral_vs_scale.png, fig3_class_persistence_heatmap.png")
