import os, json, glob
PM = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update/results/per_model"
BASE = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/followup/001-living-update"
REPL = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"

def L(safe):
    p = os.path.join(PM, safe + ".json"); return json.load(open(p)) if os.path.exists(p) else None

TIER1 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
         "EleutherAI/pythia-70m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"]
TIER2 = ["Qwen/Qwen2.5-0.5B", "HuggingFaceTB/SmolLM2-1.7B"]
NPP = {"gpt2":100,"gpt2-medium":100,"EleutherAI/pythia-70m":100,"EleutherAI/pythia-410m":100,
       "EleutherAI/pythia-1.4b":50,"gpt2-large":30,"gpt2-xl":30,"EleutherAI/pythia-2.8b":30}

per_model = {}
gpu_seconds = {}
for m in TIER1:
    d = L(m.replace("/", "__")); per_model[m] = d
    gpu_seconds[m] = round(d.get("duration_seconds", 0), 1)
for m in TIER2:
    d = L("tier2_" + m.replace("/", "__")); per_model[m] = d
    gpu_seconds[m] = round(d.get("duration_seconds", 0), 1)

# on-disk originals
repl_step2 = json.load(open(os.path.join(REPL, "step2_baseline_result.json")))
repl_step8 = json.load(open(os.path.join(REPL, "step8_completeness_result.json")))

def b(m, k):
    return per_model[m]["baseline"][k]
def f(m, k):
    return per_model[m].get("faithfulness", {}).get(k)

key_results = {
    "baseline_logit_diff": {m: round(b(m, "mean_logit_diff"), 4) for m in TIER1 + TIER2},
    "io_over_s_rate": {m: round(b(m, "io_over_s_rate"), 4) for m in TIER1 + TIER2},
    "faithfulness_FC_over_FM_full6class": {
        m: round(f(m, "F_C_over_F_M"), 4) for m in TIER1
        if len(per_model[m].get("faithfulness", {}).get("classes_included", [])) == 6},
    "gpt2xl_faithfulness_3class_partial": round(f("gpt2-xl", "F_C_over_F_M"), 4),
    "name_survivor_counts_single_token": {m: per_model[m].get("n_names_single_token") for m in TIER1},
    "tier2_name_survivor_counts": {m: per_model[m].get("n_names_single_token") for m in TIER2},
    "N_pp_per_model": NPP,
    "gpu_seconds_per_model": gpu_seconds,
    "discovered_circuit_by_class": {
        m: per_model[m].get("discovered_circuit") for m in TIER1
        if per_model[m].get("discovered_circuit")},
}

comparison = [
    {"quantity": "C2 baseline mean logit-diff (GPT-2 small)",
     "original": f"paper 3.56 / replicated {repl_step2['mean_logit_diff']:.4f}",
     "followup": f"{b('gpt2','mean_logit_diff'):.4f} (this session, N=2000)",
     "source_file": "replication/step2_baseline_result.json",
     "note": "anchor reproduced; GPT-2 family all >3.4, Pythia family fails (<=0.85, mostly <0)"},
    {"quantity": "C2 IO>S rate (GPT-2 small)",
     "original": f"paper 99.3% / replicated {repl_step2['io_over_s_rate']*100:.1f}%",
     "followup": f"{b('gpt2','io_over_s_rate')*100:.1f}%",
     "source_file": "replication/step2_baseline_result.json", "note": ""},
    {"quantity": "C3 faithfulness F(C)/F(M) (GPT-2 small)",
     "original": f"paper ~87% / replicated 87.8% (F(M)={repl_step8['faithfulness']['F_M']:.4f}, F(C)={repl_step8['faithfulness']['CIRCUIT']['F_C']:.4f})",
     "followup": f"{f('gpt2','F_C_over_F_M')*100:.1f}% (discovered 6-class circuit; paper-curated circuit via this harness = 87.78% [validate_anchor.py])",
     "source_file": "replication/step8_completeness_result.json",
     "note": "harness validated exact-match to disk on paper circuit"},
    {"quantity": "C3 faithfulness F(C)/F(M) across GPT-2 scale (discovered 6-class)",
     "original": "n/a (paper only characterises GPT-2 small)",
     "followup": f"small {f('gpt2','F_C_over_F_M')*100:.1f}% -> medium {f('gpt2-medium','F_C_over_F_M')*100:.1f}% -> large {f('gpt2-large','F_C_over_F_M')*100:.1f}%",
     "source_file": "results/per_model/*.json",
     "note": "monotone collapse of discovered-circuit faithfulness with scale"},
    {"quantity": "C3 faithfulness F(C)/F(M) (Pythia-1.4b, discovered 6-class)",
     "original": "n/a", "followup": f"{f('EleutherAI/pythia-1.4b','F_C_over_F_M')*100:.1f}% (F(M)={f('EleutherAI/pythia-1.4b','F_M'):.3f})",
     "source_file": "results/per_model/EleutherAI__pythia-1.4b.json",
     "note": "only Pythia rung with F(M)>0; other Pythia rungs fail the task so faithfulness is uninterpretable"},
    {"quantity": "C4 name-mover heads (top, h->Logits)",
     "original": "paper/replicated GPT-2 small: 9.9, 9.6, 10.0",
     "followup": "gpt2 (this session): " + ", ".join(f"{l}.{h}" for l,h in per_model['gpt2']['discovered_circuit']['name mover'][:3]) +
                 "; gpt2-large: " + ", ".join(f"{l}.{h}" for l,h in per_model['gpt2-large']['discovered_circuit']['name mover'][:3]),
     "source_file": "replication/step3_pathpatching_result.json",
     "note": "name movers persist and shift to later layers with scale"},
    {"quantity": "C9 S-inhibition heads (top, h->NM queries)",
     "original": "paper/replicated GPT-2 small: 7.3, 7.9, 8.6, 8.10",
     "followup": "gpt2 (this session): " + ", ".join(f"{l}.{h}" for l,h in per_model['gpt2']['discovered_circuit']['s2 inhibition']),
     "source_file": "replication/step3_b_h_to_NMquery_heads.npy",
     "note": "recovered on GPT-2 small; strong single S-inhibition heads vanish by gpt2-medium"},
]

summary = {
    "instruction_summary": "Test whether the IOI task behavior and circuit (Wang et al. 2022, GPT-2 small) survive up the GPT-2 scale ladder (small->XL) and across generations (Pythia 70M->2.8B; behavioral into modern small models), recomputing baseline (C2), discovery (C4/C9/C14/C16) and faithfulness (C3) per model against the on-disk replication.",
    "completed": True,
    "steps": [
        {"description": "Fresh uv venv under followup dir; editable-install replication codebase (torch 2.2.2 / transformers 4.30.2 / numpy<2)", "command": "uv venv; uv pip install -e run/replication/codebase + pinned deps", "exit_code": 0, "outputs": ["workspace/.venv"]},
        {"description": "Validate faithfulness harness reproduces on-disk anchor with paper circuit", "command": "python validate_anchor.py", "exit_code": 0, "outputs": ["F(C)/F(M)=0.8778 == disk"]},
        {"description": "Tier-1 ladder: baseline + path-patch discovery (a/b/c/d) + discovered-circuit faithfulness", "command": "python ladder_tier1.py <MODEL_ID> <configs>", "exit_code": 0, "outputs": [f"results/per_model/{m.replace('/','__')}.json" for m in TIER1]},
        {"description": "Tier-2 behavioral-only IOI (raw HF AutoModelForCausalLM, modern transformers venv)", "command": "python tier2_behavioral.py <MODEL_ID>", "exit_code": 0, "outputs": ["results/per_model/tier2_*.json"]},
        {"description": "Figures", "command": "python make_figures.py", "exit_code": 0, "outputs": ["results/faithfulness_vs_scale.png", "results/headcount_vs_scale.png", "results/behavioral_vs_scale.png"]},
    ],
    "key_results": key_results,
    "comparison_to_original": comparison,
    "artifacts": [
        "results/faithfulness_vs_scale.png", "results/headcount_vs_scale.png", "results/behavioral_vs_scale.png",
        "results/per_model/ (per-model JSON + npy discovery matrices)",
    ],
    "deviations": [
        "Instruction said Tier-1 weights were cached; only 'gpt2' was present in HF_HOME. Downloaded gpt2-medium/large/xl and all Pythia rungs fresh (HF_HUB_OFFLINE=0 for those loads). No /home writes.",
        "gpt2-xl full 6-class discovery not completed: config b alone took 38 min (N=30, 1200 heads); configs c,d exceeded budget. gpt2-xl faithfulness reported for a 3-class output-side circuit (name mover + negative + S-inhibition, 15 heads) and clearly labelled partial.",
        "Path-patch N reduced below 100 for the largest models to fit 48 GB (adaptive: N=50 for pythia-1.4b, N=30 for gpt2-large/xl and pythia-2.8b) after per-head qkv-input caching OOMed at N=100 on pythia-1.4b. Baseline (C2) always N=2000.",
        "Completeness/greedy search (C19/C21) dropped entirely as instructed (anchor-only).",
        "Induction vs duplicate-token split within config c is a layer-depth heuristic (earliest layers -> duplicate); both map to the S2 token position so the split does not affect faithfulness.",
        "Non-GPT-2 single-token name filter applied via monkeypatch of module-global NAMES (IOIDataset exposes no names= arg). Pythia/NeoX tokenizer keeps 88/99 names; all >=15 so every model was attempted.",
        "Tier-2 is behavioral-only (no circuit). OLMo-2-0425-1B could not load (arch 'olmo2' unknown to transformers 4.46.3) -> environmental skip; gated repos not probed.",
    ],
}
json.dump(summary, open(os.path.join(BASE, "followup_summary.json"), "w"), indent=2)
print("wrote followup_summary.json")
print("faithfulness 6-class:", key_results["faithfulness_FC_over_FM_full6class"])
print("baselines:", {k.split('/')[-1]: v for k, v in key_results["baseline_logit_diff"].items()})
