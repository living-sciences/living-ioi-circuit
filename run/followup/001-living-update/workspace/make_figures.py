import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PM = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update/results/per_model"
RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update/results"

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})

PARAMS = {  # millions
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558,
    "EleutherAI/pythia-70m": 70, "EleutherAI/pythia-410m": 410,
    "EleutherAI/pythia-1.4b": 1414, "EleutherAI/pythia-2.8b": 2775,
    "Qwen/Qwen2.5-0.5B": 494, "HuggingFaceTB/SmolLM2-1.7B": 1711,
}

def load(safe):
    p = os.path.join(PM, safe + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

def tload(safe):
    p = os.path.join(PM, "tier2_" + safe + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
TIER2 = ["Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-1.7B"]

D = {m: load(m.replace("/", "__")) for m in GPT2 + PYTHIA}
T = {m: tload(m.replace("/", "__")) for m in TIER2}

# ============ FIG 1: faithfulness vs scale ============
fig, axL = plt.subplots()
axR = axL.twinx()
axR.grid(False)

# faithfulness (left) — only full 6-class & interpretable (F(M)>0)
def faith_pts(models):
    xs, ys, labels = [], [], []
    for m in models:
        d = D[m]; f = d.get("faithfulness", {})
        cls = f.get("classes_included", [])
        if len(cls) == 6 and f.get("F_M", -1) > 0:
            xs.append(PARAMS[m]); ys.append(f["F_C_over_F_M"]); labels.append(m)
    return xs, ys, labels

gx, gy, gl = faith_pts(GPT2)
px, py, pl = faith_pts(PYTHIA)
axL.plot(gx, gy, "-o", color=OKABE_ITO[0], label="GPT-2 family — faithfulness (6-class)", zorder=5, ms=8)
axL.plot(px, py, "-s", color=OKABE_ITO[1], label="Pythia — faithfulness (6-class)", zorder=5, ms=8)
# gpt2-xl 3-class partial (open marker)
xl = D["gpt2-xl"]["faithfulness"]
axL.plot(PARAMS["gpt2-xl"], xl["F_C_over_F_M"], marker="o", mfc="none", mec=OKABE_ITO[0], ms=9, ls="none",
         label="gpt2-xl — 3-class (partial, c/d over budget)", zorder=6)
axL.axhline(0.87, ls="--", color="gray", lw=1, alpha=0.8)
axL.text(80, 0.885, "paper 87% / anchor 87.8%", color="gray", fontsize=9, va="bottom")

# baseline logit-diff (right) — all rungs, faint
def base_pts(models, src):
    xs, ys = [], []
    for m in models:
        b = src[m].get("baseline") if src[m] else None
        if b:
            xs.append(PARAMS[m]); ys.append(b["mean_logit_diff"])
    return xs, ys
bgx, bgy = base_pts(GPT2, D)
bpx, bpy = base_pts(PYTHIA, D)
btx, bty = base_pts(TIER2, T)
axR.plot(bgx, bgy, ":o", color=OKABE_ITO[0], alpha=0.45, ms=5, label="GPT-2 — baseline logit-diff")
axR.plot(bpx, bpy, ":s", color=OKABE_ITO[1], alpha=0.45, ms=5, label="Pythia — baseline logit-diff")
axR.plot(btx, bty, ":D", color=OKABE_ITO[2], alpha=0.6, ms=6, label="modern (Tier-2 behavioral)")
axR.axhline(0.0, color="k", lw=0.6, alpha=0.4)

axL.set_xscale("log")
axL.set_xlabel("model parameters (millions, log scale)")
axL.set_ylabel("circuit faithfulness  F(C)/F(M)")
axR.set_ylabel("baseline IOI mean logit-diff (IO − S)")
axL.set_ylim(-0.15, 1.05)
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center left", framealpha=0.9)
fig.savefig(os.path.join(RES, "faithfulness_vs_scale.png"), facecolor="white")
plt.close(fig)
print("wrote faithfulness_vs_scale.png")

# ============ FIG 2: strong-effect head-count per class vs scale ============
# The discovered circuit fixes k per class (= paper counts), so a raw head COUNT is
# constant by construction. Instead we count heads whose patching actually moves the
# logit-diff by >= THRESH (in raw logit units), which reveals which classes really exist.
# raw effect Δld = (pct_change/100) * default_logit_diff  (the normalization cancels, so
# this is stable even when the model's baseline logit-diff is near 0).
THRESH = 0.2
CLASSES = ["name mover", "negative", "s2 inhibition", "induction", "duplicate token", "previous token"]
CCOL = OKABE_ITO[:6]

def strong_counts(d):
    disc = d.get("discovery", {})
    out = {c: 0 for c in CLASSES}
    def cnt(entries, dl):
        return sum(1 for e in entries if abs(e["pct_change"] / 100.0 * dl) >= THRESH)
    a = disc.get("a_h_to_logits")
    if a:
        out["name mover"] = cnt(a["name_movers_top"], a["default_logit_diff"])
        out["negative"] = cnt(a["negNM_top"], a["default_logit_diff"])
    b = disc.get("b_h_to_NMquery")
    if b:
        out["s2 inhibition"] = cnt(b["s_inhibition_top"], b["default_logit_diff"])
    c = disc.get("c_h_to_SInhval")
    if c:
        vs = sorted(c["value_senders_top"], key=lambda x: x["layer"])
        dup, ind = vs[:3], vs[3:]
        out["duplicate token"] = cnt(dup, c["default_logit_diff"])
        out["induction"] = cnt(ind, c["default_logit_diff"])
    dd = disc.get("d_h_to_INDkey")
    if dd:
        out["previous token"] = cnt(dd["previous_token_top"], dd["default_logit_diff"])
    return out

full_models = [m for m in GPT2 + PYTHIA if len(D[m].get("faithfulness", {}).get("classes_included", [])) == 6]
full_models = sorted(full_models, key=lambda m: PARAMS[m])
labels = [m.split("/")[-1] + f"\n{PARAMS[m]}M" for m in full_models]
counts = {m: strong_counts(D[m]) for m in full_models}
fig, ax = plt.subplots(figsize=(8.5, 4.5))
bottom = np.zeros(len(full_models))
for ci, c in enumerate(CLASSES):
    vals = [counts[m][c] for m in full_models]
    ax.bar(range(len(full_models)), vals, bottom=bottom, color=CCOL[ci], label=c, edgecolor="white", lw=0.5)
    bottom += np.array(vals)
# annotate whether the model does the task
for i, m in enumerate(full_models):
    ld = D[m]["baseline"]["mean_logit_diff"]
    tag = "does IOI" if ld > 1 else ("weak" if ld > 0 else "FAILS IOI")
    ax.text(i, bottom[i] + 0.3, tag, ha="center", fontsize=8,
            color=(OKABE_ITO[2] if ld > 1 else (OKABE_ITO[1] if ld > 0 else OKABE_ITO[3])))
ax.set_xticks(range(len(full_models))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel(f"# discovered heads with |Δlogit-diff| ≥ {THRESH}")
ax.set_xlabel("model (params) — full 6-class discovery; classes are top-k by paper counts")
ax.axhline(26, ls="--", color="gray", lw=1); ax.text(-0.4, 26.3, "paper: 26 heads all effective", color="gray", fontsize=9)
ax.set_ylim(0, 28)
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.savefig(os.path.join(RES, "headcount_vs_scale.png"), facecolor="white")
plt.close(fig)
print("wrote headcount_vs_scale.png")
print("strong_counts:", {m.split('/')[-1]: counts[m] for m in full_models})

# ============ FIG 3: behavioral survival (baseline logit-diff) all families ============
fig, ax = plt.subplots()
ax.plot(bgx, bgy, "-o", color=OKABE_ITO[0], ms=8, label="GPT-2 family (small→XL)")
ax.plot(bpx, bpy, "-s", color=OKABE_ITO[1], ms=8, label="Pythia family (70M→2.8B)")
ax.plot(btx, bty, "D", color=OKABE_ITO[2], ms=9, ls="none", label="modern small (Tier-2 behavioral)")
for m, x, y in zip(TIER2, btx, bty):
    ax.annotate(m.split("/")[-1], (x, y), fontsize=8, xytext=(4, -10), textcoords="offset points")
ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("model parameters (millions, log scale)")
ax.set_ylabel("baseline IOI mean logit-diff (IO − S)")
ax.text(80, -0.75, "below 0 = model prefers the\nrepeated name (fails IOI)", fontsize=8, color=OKABE_ITO[3])
ax.legend(fontsize=9, loc="center right")
fig.savefig(os.path.join(RES, "behavioral_vs_scale.png"), facecolor="white")
plt.close(fig)
print("wrote behavioral_vs_scale.png")
