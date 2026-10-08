"""003 model-refresh tick: re-aggregate ALL (old Tier-1 + Tier-2, reused from 001) + the NEW 2026
behavioral models, and extend the over-time figure to the 2026 point.

Reads every per-model JSON from results/per_model/ (the old ones were copied verbatim from 001; the
2026 ones were produced THIS session by tier2_behavioral_2026.py). Writes:
  results/behavioral_over_time.png   (THE over-time figure, now incl. 2026)
  results/behavioral_vs_scale.png    (behavioral vs params, incl. 2026)
  results/faithfulness_vs_scale.png  (Tier-1 discovered-circuit faithfulness ladder, unchanged by 2026)
  results/aggregate_all.json         (machine-readable roll-up used by report/result_card)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/003-living-update"
PM = os.path.join(ROOT, "results", "per_model")
RES = os.path.join(ROOT, "results")

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})

# ---- model registry: params (millions), approx release year (decimal), family, tier ----
# release years are approximate, used only for the over-time x-axis (documented in report).
REG = {
    # Tier-1 (full circuit), GPT-2 family — Feb 2019
    "gpt2":                  dict(params=124,  year=2019.1, family="GPT-2",  tier=1, label="gpt2"),
    "gpt2-medium":           dict(params=355,  year=2019.1, family="GPT-2",  tier=1, label="gpt2-medium"),
    "gpt2-large":            dict(params=774,  year=2019.1, family="GPT-2",  tier=1, label="gpt2-large"),
    "gpt2-xl":               dict(params=1558, year=2019.1, family="GPT-2",  tier=1, label="gpt2-xl"),
    # Pythia family — Apr 2023
    "EleutherAI/pythia-70m": dict(params=70,   year=2023.3, family="Pythia", tier=1, label="pythia-70m"),
    "EleutherAI/pythia-410m":dict(params=410,  year=2023.3, family="Pythia", tier=1, label="pythia-410m"),
    "EleutherAI/pythia-1.4b":dict(params=1414, year=2023.3, family="Pythia", tier=1, label="pythia-1.4b"),
    "EleutherAI/pythia-2.8b":dict(params=2775, year=2023.3, family="Pythia", tier=1, label="pythia-2.8b"),
    # Tier-2 modern (behavioral), 2024
    "Qwen/Qwen2.5-0.5B":     dict(params=494,  year=2024.7, family="modern-2024", tier=2, label="Qwen2.5-0.5B"),
    "HuggingFaceTB/SmolLM2-1.7B": dict(params=1711, year=2024.9, family="modern-2024", tier=2, label="SmolLM2-1.7B"),
    # NEW: 2026 flagships (behavioral), this session
    "Qwen/Qwen3.5-9B-Base":  dict(params=9000, year=2026.2, family="2026", tier=2, label="Qwen3.5-9B-Base"),
    "Qwen/Qwen3.5-9B":       dict(params=9000, year=2026.2, family="2026", tier=2, label="Qwen3.5-9B (instruct)"),
    "google/gemma-4-12B":    dict(params=12000, year=2026.6, family="2026", tier=2, label="gemma-4-12B"),
}

def safe(m): return m.replace("/", "__")

def load_baseline(m):
    """Return (mean_logit_diff, io_over_s, io_prob, N) or None. Reads Tier-1 or Tier-2 json."""
    p1 = os.path.join(PM, safe(m) + ".json")
    p2 = os.path.join(PM, "tier2_" + safe(m) + ".json")
    path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
    if path is None:
        return None, path
    d = json.load(open(path))
    b = d.get("baseline")
    if not b:
        return None, path
    return dict(ld=b["mean_logit_diff"], io=b.get("io_over_s_rate"),
                ioprob=b.get("mean_io_prob"), N=b.get("N", b.get("N_scored"))), path

def load_faith(m):
    p1 = os.path.join(PM, safe(m) + ".json")
    if not os.path.exists(p1):
        return None
    d = json.load(open(p1))
    f = d.get("faithfulness")
    return f

agg = {}
for m, meta in REG.items():
    b, path = load_baseline(m)
    agg[m] = dict(meta=meta, baseline=b, source=os.path.relpath(path, ROOT) if path else None)
    f = load_faith(m)
    if f:
        agg[m]["faithfulness"] = f

json.dump(agg, open(os.path.join(RES, "aggregate_all.json"), "w"), indent=2, default=float)
print("wrote aggregate_all.json")

# ================= FIG A: behavioral IOI over time (THE over-time figure) =================
fam_color = {"GPT-2": OKABE_ITO[0], "Pythia": OKABE_ITO[1], "modern-2024": OKABE_ITO[2], "2026": OKABE_ITO[3]}
fam_marker = {"GPT-2": "o", "Pythia": "s", "modern-2024": "D", "2026": "*"}
# per-model label offsets (x_pts, y_pts, ha) to avoid overlaps in dense clusters
LBL = {
    "gpt2": (14, -14, "left"), "gpt2-medium": (14, 2, "left"),
    "gpt2-large": (0, 8, "center"), "gpt2-xl": (-14, 2, "right"),
    "pythia-1.4b": (0, 8, "center"), "pythia-70m": (-6, -14, "right"),
    "pythia-410m": (0, -16, "center"), "pythia-2.8b": (6, -14, "left"),
    "Qwen2.5-0.5B": (0, 8, "center"), "SmolLM2-1.7B": (0, -14, "center"),
    "Qwen3.5-9B-Base": (-8, -16, "right"), "Qwen3.5-9B (instruct)": (-8, 10, "right"),
    "gemma-4-12B": (10, -2, "left"),
}
fig, ax = plt.subplots(figsize=(9.0, 5.2))
for fam in ["GPT-2", "Pythia", "modern-2024", "2026"]:
    xs, ys, labels = [], [], []
    for m, a in agg.items():
        if a["meta"]["family"] != fam or a["baseline"] is None:
            continue
        xs.append(a["meta"]["year"]); ys.append(a["baseline"]["ld"]); labels.append(a["meta"]["label"])
    ms = 20 if fam == "2026" else (9 if fam == "modern-2024" else 8)
    ax.plot(xs, ys, fam_marker[fam], color=fam_color[fam], ms=ms, ls="none",
            label=fam, mec="black" if fam == "2026" else None, mew=0.6 if fam == "2026" else 0)
    for x, y, lab in zip(xs, ys, labels):
        dx, dy, ha = LBL.get(lab, (0, 8, "center"))
        ax.annotate(lab, (x, y), fontsize=7.5, xytext=(dx, dy), textcoords="offset points",
                    ha=ha, color=fam_color[fam])
ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
ax.axhline(3.49, ls="--", color="gray", lw=1, alpha=0.8)
ax.text(2023.2, 3.58, "GPT-2 small anchor 3.49 (replication)", color="gray", fontsize=8.5, va="bottom", ha="center")
ax.set_xlabel("approximate model release year")
ax.set_ylabel("baseline IOI mean logit-diff (IO − S)")
ax.set_title("IOI behaviour over time: still holds — stronger — in 2026 models")
ax.text(2020.4, 1.2, "below 0 = model prefers the\nrepeated name (fails IOI)", fontsize=8, color=OKABE_ITO[3])
ax.annotate("", xy=(2023.0, -0.3), xytext=(2021.3, 1.05),
            arrowprops=dict(arrowstyle="->", color=OKABE_ITO[3], lw=1, alpha=0.7))
ax.set_xticks([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026])
ax.set_ylim(-1.1, 6.6)
ax.legend(fontsize=9, loc="center left")
fig.savefig(os.path.join(RES, "behavioral_over_time.png"), facecolor="white")
plt.close(fig)
print("wrote behavioral_over_time.png")

# ================= FIG B: behavioral vs scale (params), incl. 2026 =================
fig, ax = plt.subplots(figsize=(8.0, 4.8))
for fam in ["GPT-2", "Pythia", "modern-2024", "2026"]:
    xs, ys, labels = [], [], []
    for m, a in agg.items():
        if a["meta"]["family"] != fam or a["baseline"] is None:
            continue
        xs.append(a["meta"]["params"]); ys.append(a["baseline"]["ld"]); labels.append(a["meta"]["label"])
    order = np.argsort(xs)
    xs = list(np.array(xs)[order]); ys = list(np.array(ys)[order]); labels = list(np.array(labels)[order])
    ms = 18 if fam == "2026" else 8
    ls = "-" if fam in ("GPT-2", "Pythia") else "none"
    ax.plot(xs, ys, fam_marker[fam], color=fam_color[fam], ms=ms, ls=ls,
            label=fam, mec="black" if fam == "2026" else None, mew=0.6 if fam == "2026" else 0)
    if fam in ("modern-2024", "2026"):
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=7.5, xytext=(4, -11 if fam=="modern-2024" else 9),
                        textcoords="offset points", color=fam_color[fam])
ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("model parameters (millions, log scale)")
ax.set_ylabel("baseline IOI mean logit-diff (IO − S)")
ax.set_title("IOI behaviour vs scale, GPT-2 → 2026 flagships")
ax.legend(fontsize=9, loc="center right")
fig.savefig(os.path.join(RES, "behavioral_vs_scale.png"), facecolor="white")
plt.close(fig)
print("wrote behavioral_vs_scale.png")

# ================= FIG C: Tier-1 faithfulness ladder (unchanged by 2026; regenerated for self-containedness) =================
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
fig, axL = plt.subplots(figsize=(8.0, 4.8))
def faith_pts(models):
    xs, ys = [], []
    for m in models:
        f = agg[m].get("faithfulness", {})
        if f and len(f.get("classes_included", [])) == 6 and f.get("F_M", -1) > 0:
            xs.append(REG[m]["params"]); ys.append(f["F_C_over_F_M"])
    return xs, ys
gx, gy = faith_pts(GPT2); px, py = faith_pts(PYTHIA)
axL.plot(gx, gy, "-o", color=OKABE_ITO[0], ms=8, label="GPT-2 family — faithfulness (6-class)")
axL.plot(px, py, "-s", color=OKABE_ITO[1], ms=8, label="Pythia — faithfulness (6-class)")
xl = agg["gpt2-xl"].get("faithfulness")
if xl:
    axL.plot(REG["gpt2-xl"]["params"], xl["F_C_over_F_M"], marker="o", mfc="none", mec=OKABE_ITO[0],
             ms=9, ls="none", label="gpt2-xl — 3-class (partial)")
axL.axhline(0.87, ls="--", color="gray", lw=1, alpha=0.8)
axL.text(80, 0.885, "paper 87% / anchor 87.8%", color="gray", fontsize=9, va="bottom")
axL.set_xscale("log")
axL.set_xlabel("model parameters (millions, log scale)")
axL.set_ylabel("discovered-circuit faithfulness  F(C)/F(M)")
axL.set_title("Circuit faithfulness ladder (Tier-1; 2026 models are behavioral-only)")
axL.set_ylim(-0.15, 1.05)
axL.legend(fontsize=8.5, loc="upper right")
fig.savefig(os.path.join(RES, "faithfulness_vs_scale.png"), facecolor="white")
plt.close(fig)
print("wrote faithfulness_vs_scale.png")

# ---- print roll-up table ----
print("\n=== behavioral roll-up (all models) ===")
rows = sorted(agg.items(), key=lambda kv: (kv[1]["meta"]["year"], kv[1]["meta"]["params"]))
for m, a in rows:
    b = a["baseline"]
    if b is None: continue
    print(f"{a['meta']['label']:<22} {a['meta']['params']:>6}M  yr{a['meta']['year']:<7} "
          f"ld={b['ld']:+.3f}  IO>S={b['io']:.3f}  ioprob={b['ioprob']:.3f}  tier{a['meta']['tier']}")
