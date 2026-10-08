import os, json, time, random
from copy import deepcopy
import torch
import numpy as np
from tqdm import tqdm

from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import logit_diff, basis_change, get_heads_from_nodes
from easy_transformer.ioi_circuit_extraction import (
    CIRCUIT, NAIVE, RELEVANT_TOKENS, get_heads_circuit, do_circuit_extraction,
)

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
SETS = os.path.join(OUT, "sets"); os.makedirs(SETS, exist_ok=True)
PTS = os.path.join(OUT, "pts"); os.makedirs(PTS, exist_ok=True)
torch.set_grad_enabled(False)
random.seed(0); torch.manual_seed(0)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

N = 100
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
abc_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)
mean_dataset = abc_dataset

def get_all_nodes(circuit):
    nodes = []
    for c in circuit:
        for head in circuit[c]:
            nodes.append((head, RELEVANT_TOKENS[head][0]))
    return nodes

summary = {"N": N}

# ================= PART A: faithfulness F(M), F(C) =================
model.reset_hooks()
F_M = logit_diff(model, ioi_dataset).item()
print("F(M) logit_diff_M =", F_M, flush=True)
faith = {"F_M": F_M}
for name, circ in [("CIRCUIT", CIRCUIT), ("NAIVE", NAIVE)]:
    heads_to_keep = get_heads_circuit(ioi_dataset, excluded=[], circuit=circ.copy())
    model.reset_hooks()
    model, _ = do_circuit_extraction(model=model, heads_to_keep=heads_to_keep,
                                     mlps_to_remove={}, ioi_dataset=ioi_dataset, mean_dataset=mean_dataset)
    F_C = logit_diff(model, ioi_dataset).item()
    model.reset_hooks()
    faith[name] = {"F_C": F_C, "abs_F_M_minus_F_C": abs(F_M - F_C)}
    print(f"  {name}: F(C)={F_C:.4f}  |F(M)-F(C)|={abs(F_M-F_C):.4f}", flush=True)
summary["faithfulness"] = faith

# ================= PART B: by-class completeness sweep (both circuits) =================
def by_class_sweep(circuit, tag):
    perf_by_sets = []
    circuit_perf = []
    for G in tqdm(list(circuit.keys()) + ["none"]):
        excluded_classes = [] if G == "none" else [G]
        # F(C\G): keep circuit minus class G
        heads_to_keep = get_heads_circuit(ioi_dataset, excluded=excluded_classes, circuit=circuit)
        model.reset_hooks()
        model2, _ = do_circuit_extraction(model=model, heads_to_keep=heads_to_keep,
                                          mlps_to_remove={}, ioi_dataset=ioi_dataset, mean_dataset=mean_dataset)
        broken, _ = logit_diff(model2, ioi_dataset, std=True, all=True)
        # F(M\G): remove ONLY class G's heads from the full model
        excl_class = list(circuit.keys())
        if G != "none":
            excl_class.remove(G)
        G_heads_to_remove = get_heads_circuit(ioi_dataset, excluded=excl_class, circuit=circuit)
        model.reset_hooks()
        model2, _ = do_circuit_extraction(model=model, heads_to_remove=G_heads_to_remove,
                                          mlps_to_remove={}, ioi_dataset=ioi_dataset, mean_dataset=mean_dataset)
        cobble, _ = logit_diff(model2, ioi_dataset, std=True, all=True)
        model.reset_hooks()
        on_d, off_d = [], []
        for i in range(len(cobble)):
            x, y = basis_change(float(broken[i]), float(cobble[i]))
            on_d.append(x); off_d.append(y)
            circuit_perf.append({"removed_set_id": G, "ldiff_broken": float(broken[i]),
                                 "ldiff_cobble": float(cobble[i])})
        entry = {
            "removed_group": G,
            "mean_cur_metric_broken_F_C_minus_K": float(broken.mean()),
            "mean_cur_metric_cobble_F_M_minus_K": float(cobble.mean()),
            "std_broken": float(broken.std()), "std_cobble": float(cobble.std()),
            "mean_abs_diff_incompleteness": float(abs(broken.mean() - cobble.mean())),
        }
        perf_by_sets.append(entry)
        print(f"    [{tag}] G={G:16s} F(C\\K)={entry['mean_cur_metric_broken_F_C_minus_K']:.3f} "
              f"F(M\\K)={entry['mean_cur_metric_cobble_F_M_minus_K']:.3f} "
              f"|diff|={entry['mean_abs_diff_incompleteness']:.3f}", flush=True)
    with open(os.path.join(SETS, f"perf_by_classes_{tag}.json"), "w") as f:
        json.dump({"perf_by_sets": perf_by_sets, "per_prompt": circuit_perf}, f, indent=2)
    return perf_by_sets

summary["by_class"] = {}
for tag, circ in [("complete", deepcopy(CIRCUIT)), ("naive", deepcopy(NAIVE))]:
    print(f"=== by-class sweep: {tag} ===", flush=True)
    summary["by_class"][tag] = by_class_sweep(circ, tag)
    summary["by_class"][tag + "_max_incompleteness"] = max(e["mean_abs_diff_incompleteness"] for e in summary["by_class"][tag])

# ================= PART C: greedy + random search (both circuits) =================
def build_evals(circuit):
    all_nodes = get_all_nodes(circuit)
    all_circuit_nodes = [h[0] for h in all_nodes]
    circuit_size = len(all_circuit_nodes)
    complement_hooks = do_circuit_extraction(model=model, heads_to_keep={}, mlps_to_remove={},
                                             ioi_dataset=ioi_dataset, mean_dataset=mean_dataset,
                                             return_hooks=True, hooks_dict=True)
    assert len(complement_hooks) == 144
    heads_to_keep = get_heads_from_nodes(all_nodes, ioi_dataset)
    circuit_hooks = do_circuit_extraction(model=model, heads_to_keep=heads_to_keep, mlps_to_remove={},
                                          ioi_dataset=ioi_dataset, mean_dataset=mean_dataset,
                                          return_hooks=True, hooks_dict=True)
    model_rem_hooks = do_circuit_extraction(model=model, heads_to_remove=heads_to_keep, mlps_to_remove={},
                                            ioi_dataset=ioi_dataset, mean_dataset=mean_dataset,
                                            return_hooks=True, hooks_dict=True)
    for (layer, head_idx) in list(circuit_hooks.keys()):
        if (layer, head_idx) not in heads_to_keep.keys():
            circuit_hooks.pop((layer, head_idx))
    assert len(circuit_hooks) == circuit_size

    def cobble_eval(nodes):
        model.reset_hooks()
        for head in nodes:
            model.add_hook(*model_rem_hooks[head])
        ld = logit_diff(model, ioi_dataset)
        model.reset_hooks()
        return ld

    def circuit_eval(nodes):
        model.reset_hooks()
        for head in all_circuit_nodes:
            if head not in nodes:
                model.add_hook(*circuit_hooks[head])
        for head in complement_hooks:
            if head not in all_circuit_nodes or head in nodes:
                model.add_hook(*complement_hooks[head])
        ld = logit_diff(model, ioi_dataset)
        model.reset_hooks()
        return ld

    return all_nodes, all_circuit_nodes, circuit_size, circuit_eval, cobble_eval


def greedy_search(circuit, mode, no_runs=10, no_iters=10):
    all_nodes, all_circuit_nodes, circuit_size, circuit_eval, cobble_eval = build_evals(circuit)
    no_samples = 10 if circuit_size == 26 else 5
    C_init = [h[0] for h in all_nodes]
    baseline = torch.abs(circuit_eval([]) - cobble_eval([]))
    xs, ys = [], []   # accepted (ceval, meval) points across all runs
    run_finals = []
    for run in tqdm(range(no_runs)):
        C_minus_G = deepcopy(C_init); G = []
        old_diff = baseline.clone()
        last_c, last_m = None, None
        for it in range(no_iters):
            to_test = random.sample(C_minus_G, min(no_samples, len(C_minus_G)))
            cevals, mevals, results = [], [], []
            for node in to_test:
                Gp = deepcopy(G) + [node]
                cevals.append(circuit_eval(Gp).item())
                mevals.append(cobble_eval(Gp).item())
                results.append(abs(cevals[-1] - mevals[-1]))
            bi = int(np.argmax(results))
            if results[bi] > old_diff:
                C_minus_G.remove(to_test[bi]); G.append(to_test[bi])
                old_diff = results[bi]
                last_c, last_m = cevals[bi], mevals[bi]
                xs.append(cevals[bi]); ys.append(mevals[bi])
        if last_c is not None:
            run_finals.append({"ceval": last_c, "meval": last_m, "diff": float(old_diff)})
    torch.save(xs, os.path.join(PTS, f"{mode}_xs.pt"))
    torch.save(ys, os.path.join(PTS, f"{mode}_ys.pt"))
    max_inc = max((abs(x - y) for x, y in zip(xs, ys)), default=0.0)
    print(f"[greedy {mode}] {len(xs)} accepted pts, baseline_incompleteness={float(baseline):.3f}, "
          f"max greedy incompleteness={max_inc:.3f}", flush=True)
    return {"n_points": len(xs), "baseline_incompleteness": float(baseline),
            "max_incompleteness": float(max_inc), "run_finals": run_finals,
            "circuit_eval": circuit_eval, "cobble_eval": cobble_eval, "all_nodes": all_nodes}


def random_search(mode, all_nodes, circuit_eval, cobble_eval, n=100):
    xs, ys = [], []
    for _ in range(n):
        indicator = torch.randint(0, 2, (len(all_nodes),))
        nodes = [node[0] for node, ind in zip(all_nodes, indicator) if ind == 1]
        xs.append(float(circuit_eval(nodes)))
        ys.append(float(cobble_eval(nodes)))
    torch.save(xs, os.path.join(PTS, f"{mode}_random_xs.pt"))
    torch.save(ys, os.path.join(PTS, f"{mode}_random_ys.pt"))
    max_inc = max(abs(x - y) for x, y in zip(xs, ys))
    print(f"[random {mode}] {n} pts, max random incompleteness={max_inc:.3f}", flush=True)
    return {"n_points": n, "max_incompleteness": float(max_inc)}

summary["greedy"] = {}
summary["random"] = {}
for mode, circ in [("complete", deepcopy(CIRCUIT)), ("naive", deepcopy(NAIVE))]:
    print(f"=== greedy+random: {mode} ===", flush=True)
    g = greedy_search(circ, mode)
    summary["random"][mode] = random_search(mode, g["all_nodes"], g["circuit_eval"], g["cobble_eval"])
    summary["greedy"][mode] = {k: g[k] for k in ["n_points", "baseline_incompleteness", "max_incompleteness", "run_finals"]}

summary["duration_seconds"] = time.time() - t0
# strip non-serializable
with open(os.path.join(OUT, "step8_completeness_result.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("DONE in %.1fs" % (time.time() - t0), flush=True)
