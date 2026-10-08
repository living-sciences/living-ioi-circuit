import os, json, time
from copy import deepcopy
import torch
import random as rd
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import probs, logit_diff, show_tokens

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
torch.set_grad_enabled(False)
rd.seed(0)
t0 = time.time()

model = EasyTransformer.from_pretrained("gpt2").cuda()
model.set_use_attn_result(True)

ADX_TEMPLATE = [
    " [A] had a good day.",
    " [A] was enjoying the situation.",
    " [A] was tired.",
    " [A] enjoyed being with a friend.",
    " [A] was an enthusiast person.",
]
DOUBLE_ADX_TEMPLATE = [x1 + x2 for x1 in ADX_TEMPLATE for x2 in ADX_TEMPLATE]


# BOS-free tokenization matching how IOIDataset.word_idx was computed.
# (Shipped gen_adv used show_tokens/to_tokens, which in this version prepends a
#  BOS token, shifting every index by 1 and breaking the punct-position assert.)
def toks_list(text):
    return [model.tokenizer.decode(t) for t in model.tokenizer(text)["input_ids"]]


def gen_adv(ioi_dataset, model, templates, name="IO"):
    adv_ioi_dataset = deepcopy(ioi_dataset)
    for i, s in enumerate(ioi_dataset.sentences):
        adv_temp = rd.choice(templates)
        adv_temp = adv_temp.replace("[A]", ioi_dataset.ioi_prompts[i][name])
        adv_tok_len = len(toks_list(adv_temp))
        punct_idx = int(ioi_dataset.word_idx["punct"][i])
        txt_toks = toks_list(s)
        punct_str_idx = len("".join(txt_toks[:punct_idx]))
        assert s[punct_str_idx] in [".", ","], f"{s} --- {s[punct_str_idx]} -- {i}"
        s = s[: punct_str_idx + 1] + adv_temp + s[punct_str_idx + 1 :]
        adv_ioi_dataset.ioi_prompts[i]["text"] = s
        adv_ioi_dataset.sentences[i] = s
        adv_ioi_dataset.word_idx["end"][i] += adv_tok_len
        adv_ioi_dataset.word_idx["S2"][i] += adv_tok_len
        adv_ioi_dataset.toks = torch.tensor(
            model.tokenizer(adv_ioi_dataset.sentences, padding=True)["input_ids"])
    return adv_ioi_dataset


N = 500
ioi_dataset = IOIDataset(prompt_type="mixed", N=N, tokenizer=model.tokenizer)
print("building adversarial (added-IO) dataset ...", flush=True)
adv_IO = gen_adv(ioi_dataset, model, DOUBLE_ADX_TEMPLATE, name="IO")
print("building control (added-S) dataset ...", flush=True)
adv_S = gen_adv(ioi_dataset, model, DOUBLE_ADX_TEMPLATE, name="S")

def metrics(ds):
    ld = logit_diff(model, ds, all=True)
    io_p = probs(model, ds, all=True, type="io")
    return {
        "mean_logit_diff": float(ld.mean()),
        "mean_io_prob": float(io_p.mean()),
        "prop_S_logit_gt_IO_logit": float((ld < 0).float().mean()),
    }

results = {
    "N": N,
    "example_adv_IO_sentence": adv_IO.sentences[0],
    "example_control_S_sentence": adv_S.sentences[0],
    "pIOI": metrics(ioi_dataset),
    "control_added_S": metrics(adv_S),
    "adversarial_added_IO": metrics(adv_IO),
    "duration_seconds": time.time() - t0,
}
print(json.dumps({k: results[k] for k in ["pIOI", "control_added_S", "adversarial_added_IO"]}, indent=2), flush=True)
print("adv IO example:", results["example_adv_IO_sentence"], flush=True)
with open(os.path.join(OUT, "step10_advex_result.json"), "w") as f:
    json.dump(results, f, indent=2)
print("DONE", flush=True)
