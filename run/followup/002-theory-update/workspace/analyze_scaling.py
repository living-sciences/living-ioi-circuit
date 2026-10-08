#!/usr/bin/env python3
"""002-theory-update: scale/generation analysis over 001's on-disk artifacts.

Reads only followup/001-living-update/results/per_model/*.json (no GPU, no recompute
of originals). Builds (1) class-persistence matrices, (2) simple scale fits with R2,
(3) generation contrast, (4) behavioral-only Tier-2 rung. Writes results/fit_results.json.
"""
import json, os
import numpy as np
from scipy import stats

R001 = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update/results/per_model"
OUT  = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/002-theory-update/results"
os.makedirs(OUT, exist_ok=True)

# Canonical total parameter counts (millions), incl. embeddings.
PARAMS_M = {
    "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558,
    "EleutherAI/pythia-70m": 70, "EleutherAI/pythia-410m": 410,
    "EleutherAI/pythia-1.4b": 1400, "EleutherAI/pythia-2.8b": 2800,
    "Qwen/Qwen2.5-0.5B": 494, "HuggingFaceTB/SmolLM2-1.7B": 1710,
}
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
PYTHIA = ["EleutherAI/pythia-70m", "EleutherAI/pythia-410m",
          "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
TIER1 = GPT2 + PYTHIA
FILES = {
    "gpt2": "gpt2.json", "gpt2-medium": "gpt2-medium.json",
    "gpt2-large": "gpt2-large.json", "gpt2-xl": "gpt2-xl.json",
    "EleutherAI/pythia-70m": "EleutherAI__pythia-70m.json",
    "EleutherAI/pythia-410m": "EleutherAI__pythia-410m.json",
    "EleutherAI/pythia-1.4b": "EleutherAI__pythia-1.4b.json",
    "EleutherAI/pythia-2.8b": "EleutherAI__pythia-2.8b.json",
}
TIER2_FILES = {
    "Qwen/Qwen2.5-0.5B": "tier2_Qwen__Qwen2.5-0.5B.json",
    "HuggingFaceTB/SmolLM2-1.7B": "tier2_HuggingFaceTB__SmolLM2-1.7B.json",
}
CLASSES = ["name mover", "s2 inhibition", "induction",
           "duplicate token", "previous token", "negative"]
STRONG_THR = 0.2  # |Delta logit-diff| threshold used by 001's card

data = {m: json.load(open(os.path.join(R001, FILES[m]))) for m in TIER1}
tier2 = {m: json.load(open(os.path.join(R001, TIER2_FILES[m]))) for m in TIER2_FILES}

# ---- which configs were actually run per model (a=name/neg, b=s-inhib, c=ind/dup, d=prev) ----
def configs_run(d):
    disc = d.get("discovery", {})
    return {"a": "a_h_to_logits" in disc, "b": "b_h_to_NMquery" in disc,
            "c": "c_h_to_SInhval" in disc, "d": "d_h_to_INDkey" in disc}

CLASS_CFG = {"name mover": "a", "negative": "a", "s2 inhibition": "b",
             "induction": "c", "duplicate token": "c", "previous token": "d"}

# ---- (1a) literal "discovered circuit" template membership matrix ----
template_matrix = {}
for m in TIER1:
    counts = data[m].get("discovered_circuit_counts", {})
    template_matrix[m] = {c: counts.get(c, 0) for c in CLASSES}

# ---- (1b) strong-head matrix: |pct/100 * default_ld| >= 0.2, per class ----
def strong_matrix_row(d):
    disc = d["discovery"]
    ld = abs(disc["a_h_to_logits"]["default_logit_diff"])
    cr = configs_run(d)
    row = {}
    def count_strong(lst):
        return int(sum(1 for h in lst if abs(h["pct_change"]) / 100.0 * ld >= STRONG_THR))
    # name mover / negative (config a)
    row["name mover"] = count_strong(disc["a_h_to_logits"]["name_movers_top"]) if cr["a"] else None
    row["negative"]   = count_strong(disc["a_h_to_logits"].get("negNM_top", [])) if cr["a"] else None
    # s2 inhibition (config b)
    row["s2 inhibition"] = count_strong(disc["b_h_to_NMquery"]["s_inhibition_top"]) if cr["b"] else None
    # induction + duplicate (config c value senders, split via 001's discovered assignment)
    if cr["c"]:
        vs = disc["c_h_to_SInhval"]["value_senders_top"]
        pct = {(h["layer"], h["head"]): h["pct_change"] for h in vs}
        dbc = data_by_class[d["model_id"]]
        def count_split(cls):
            n = 0
            for lay, hd in dbc.get(cls, []):
                p = pct.get((lay, hd))
                if p is not None and abs(p) / 100.0 * ld >= STRONG_THR:
                    n += 1
            return n
        row["induction"] = count_split("induction")
        row["duplicate token"] = count_split("duplicate token")
    else:
        row["induction"] = None
        row["duplicate token"] = None
    # previous token (config d)
    row["previous token"] = count_strong(disc["d_h_to_INDkey"]["previous_token_top"]) if cr["d"] else None
    return row

# discovered_circuit_by_class lives in the 001 followup_summary.json
summ = json.load(open(os.path.join(os.path.dirname(R001), "..",
        "followup_summary.json")))
data_by_class = summ["key_results"]["discovered_circuit_by_class"]
# normalise to (layer,head) tuples
for m in data_by_class:
    for c in data_by_class[m]:
        data_by_class[m][c] = [tuple(x) for x in data_by_class[m][c]]

strong_matrix = {m: strong_matrix_row(data[m]) for m in TIER1}

# ---- helper: OLS fit y ~ log10(params) ----
def fit(models, yvals):
    x = np.array([np.log10(PARAMS_M[m]) for m in models], float)
    y = np.array(yvals, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return {"n": int(len(x)), "note": "n<2, no fit"}
    res = stats.linregress(x, y)
    return {"n": int(len(x)), "slope_per_decade": float(res.slope),
            "intercept": float(res.intercept), "r2": float(res.rvalue**2),
            "p_value": float(res.pvalue), "std_err": float(res.stderr)}

# ---- pull scalar series ----
baseline_ld = {m: data[m]["baseline"]["mean_logit_diff"] for m in TIER1}
io_over_s   = {m: data[m]["baseline"]["io_over_s_rate"] for m in TIER1}
FM = {m: data[m]["faithfulness"]["F_M"] for m in TIER1}
FC = {m: data[m]["faithfulness"]["F_C"] for m in TIER1}
faith = {m: data[m]["faithfulness"]["F_C_over_F_M"] for m in TIER1}
n_pp = {m: data[m]["discovery"]["N"] for m in TIER1}
total_heads = {m: data[m]["discovered_circuit_total_heads"] for m in TIER1}

for m in TIER2_FILES:
    baseline_ld[m] = tier2[m]["baseline"]["mean_logit_diff"]
    io_over_s[m]   = tier2[m]["baseline"]["io_over_s_rate"]

# faithfulness is only interpretable where the model does the task (F_M > 0)
faith_ok_models = [m for m in TIER1 if FM[m] > 0]

# ---- (2) fits ----
fits = {}
# faithfulness vs log-params
fits["faithfulness_vs_logparams_gpt2_FMpos"] = fit(
    [m for m in GPT2 if FM[m] > 0], [faith[m] for m in GPT2 if FM[m] > 0])
fits["faithfulness_vs_logparams_all_FMpos"] = fit(
    faith_ok_models, [faith[m] for m in faith_ok_models])
# circuit head-count (template) vs log-params  -- expected degenerate
fits["headcount_template_vs_logparams_full6class"] = fit(
    [m for m in TIER1 if data[m]["discovered_circuit_num_classes"] == 6],
    [total_heads[m] for m in TIER1 if data[m]["discovered_circuit_num_classes"] == 6])
# strong-head total (genuine) vs log-params, only rungs with all abcd run
full_abcd = [m for m in TIER1 if all(configs_run(data[m]).values())]
strong_total = {m: int(sum(v for v in strong_matrix[m].values() if v is not None)) for m in full_abcd}
fits["strong_headtotal_vs_logparams_abcd"] = fit(full_abcd, [strong_total[m] for m in full_abcd])
# name-mover count vs log-params: template (const) and strong
fits["namemover_template_vs_logparams"] = fit(
    [m for m in TIER1 if configs_run(data[m])["a"]],
    [template_matrix[m]["name mover"] for m in TIER1 if configs_run(data[m])["a"]])
fits["namemover_strong_vs_logparams"] = fit(
    [m for m in TIER1 if strong_matrix[m]["name mover"] is not None],
    [strong_matrix[m]["name mover"] for m in TIER1 if strong_matrix[m]["name mover"] is not None])
# s-inhibition count vs log-params: template and strong
fits["sinhib_template_vs_logparams"] = fit(
    [m for m in TIER1 if configs_run(data[m])["b"]],
    [template_matrix[m]["s2 inhibition"] for m in TIER1 if configs_run(data[m])["b"]])
fits["sinhib_strong_vs_logparams"] = fit(
    [m for m in TIER1 if strong_matrix[m]["s2 inhibition"] is not None],
    [strong_matrix[m]["s2 inhibition"] for m in TIER1 if strong_matrix[m]["s2 inhibition"] is not None])
# behavioral: logit-diff vs log-params (GPT-2 only; Tier1+Tier2 excluding Pythia-fail; all)
fits["logitdiff_vs_logparams_gpt2"] = fit(GPT2, [baseline_ld[m] for m in GPT2])
fits["logitdiff_vs_logparams_pythia"] = fit(PYTHIA, [baseline_ld[m] for m in PYTHIA])
behav_all = GPT2 + PYTHIA + list(TIER2_FILES)
fits["logitdiff_vs_logparams_all_incl_tier2"] = fit(behav_all, [baseline_ld[m] for m in behav_all])

# ---- (3) generation contrast at matched scale ----
gen_contrast = {
    "gpt2_family_logitdiff": {m: round(baseline_ld[m], 3) for m in GPT2},
    "pythia_family_logitdiff": {m: round(baseline_ld[m], 3) for m in PYTHIA},
    "gpt2_family_io_over_s": {m: round(io_over_s[m], 4) for m in GPT2},
    "pythia_family_io_over_s": {m: round(io_over_s[m], 4) for m in PYTHIA},
    "matched_scale_pairs": [
        {"gpt2": "gpt2-large (774M)", "ld": round(baseline_ld["gpt2-large"], 3),
         "pythia": "pythia-1.4b (1.4B)", "pythia_ld": round(baseline_ld["EleutherAI/pythia-1.4b"], 3)},
        {"gpt2": "gpt2-xl (1.5B)", "ld": round(baseline_ld["gpt2-xl"], 3),
         "pythia": "pythia-1.4b (1.4B)", "pythia_ld": round(baseline_ld["EleutherAI/pythia-1.4b"], 3)},
    ],
    "faithfulness_matched": {
        "gpt2-small (124M)": round(faith["gpt2"], 4),
        "pythia-1.4b (1.4B, only FM>0 Pythia)": round(faith["EleutherAI/pythia-1.4b"], 4),
    },
}

# ---- assemble ----
out = {
    "params_M": PARAMS_M,
    "series": {
        "baseline_logit_diff": {m: round(v, 4) for m, v in baseline_ld.items()},
        "io_over_s_rate": {m: round(v, 4) for m, v in io_over_s.items()},
        "F_M": {m: round(FM[m], 4) for m in TIER1},
        "F_C": {m: round(FC[m], 4) for m in TIER1},
        "faithfulness_FC_over_FM": {m: round(faith[m], 4) for m in TIER1},
        "faithfulness_interpretable_FM_gt0": {m: (FM[m] > 0) for m in TIER1},
        "N_pathpatch": n_pp,
        "total_heads_template": total_heads,
        "num_classes_discovered": {m: data[m]["discovered_circuit_num_classes"] for m in TIER1},
        "configs_run": {m: configs_run(data[m]) for m in TIER1},
    },
    "persistence_matrix_template": template_matrix,
    "persistence_matrix_strong_heads": strong_matrix,
    "strong_head_total_abcd": strong_total,
    "fits": fits,
    "generation_contrast": gen_contrast,
    "tier2_behavioral": {m: {"logit_diff": round(baseline_ld[m], 4),
                             "io_over_s": round(io_over_s[m], 4)} for m in TIER2_FILES},
    "notes": {
        "strong_threshold": "|pct_change/100 * default_logit_diff| >= 0.2 (matches 001 card)",
        "template_note": "discovered_circuit_counts is a FIXED per-class top-k template (11/2/4/3/4/2=26); it does not vary across models, so total head-count and per-class counts are constant by construction where configs a-d all ran.",
        "coverage_note": "gpt2-xl ran only configs a,b (3 classes, 17 heads); pythia-2.8b only config a (2 classes, 13 heads) -- 001 budget deviations, NOT true class absence. Those cells are null (not-run), distinct from 0 (run, no strong head).",
        "faithfulness_note": "F_C/F_M is only interpretable where F_M>0 (model does IOI). Pythia 70m/410m/2.8b have F_M<0 so their ratios are excluded from fits.",
    },
}
with open(os.path.join(OUT, "fit_results.json"), "w") as f:
    json.dump(out, f, indent=2)

# ---- console summary ----
print("== strong-head persistence matrix (rows=class, None=config-not-run) ==")
hdr = ["class"] + [m.split("/")[-1] for m in TIER1]
print("  ".join(f"{h:>14}" for h in hdr))
for c in CLASSES:
    row = [c] + [("--" if strong_matrix[m][c] is None else str(strong_matrix[m][c])) for m in TIER1]
    print("  ".join(f"{str(x):>14}" for x in row))
print("\n== template matrix total heads ==", {m.split('/')[-1]: total_heads[m] for m in TIER1})
print("\n== KEY FITS ==")
for k, v in fits.items():
    print(f"{k}: {v}")
print("\n== faithfulness (FM>0 only) ==", {m: round(faith[m],3) for m in faith_ok_models})
print("== generation contrast matched pairs ==", gen_contrast["matched_scale_pairs"])
print("\nWrote", os.path.join(OUT, "fit_results.json"))
