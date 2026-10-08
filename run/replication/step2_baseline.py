import os, json, time
import torch
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset

OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/ioi-gpt2-circuit/run/replication"
t0 = time.time()
torch.set_grad_enabled(False)

m = EasyTransformer.from_pretrained("gpt2").cuda()
m.set_use_attn_result(True)

N = 100000
print(f"Building IOIDataset prompt_type='mixed' N={N} ...", flush=True)
d = IOIDataset(prompt_type="mixed", N=N, tokenizer=m.tokenizer, prepend_bos=False)
print("Dataset built in %.1fs, toks shape %s" % (time.time()-t0, tuple(d.toks.shape)), flush=True)

toks = d.toks.long().cuda()
end_idx = torch.as_tensor(d.word_idx["end"]).long()
io_ids = torch.as_tensor(d.io_tokenIDs).long()
s_ids = torch.as_tensor(d.s_tokenIDs).long()

bs = 250
ld_all = torch.empty(N)
ioprob_all = torch.empty(N)
for i in range(0, N, bs):
    j = min(i+bs, N)
    logits = m(toks[i:j]).detach()
    ar = torch.arange(j-i)
    e = end_idx[i:j]
    end_logits = logits[ar, e, :]                      # (b, vocab)
    io_l = end_logits[ar, io_ids[i:j]]
    s_l = end_logits[ar, s_ids[i:j]]
    ld_all[i:j] = (io_l - s_l).cpu()
    probs = torch.softmax(end_logits, dim=1)
    ioprob_all[i:j] = probs[ar, io_ids[i:j]].cpu()
    if (i // bs) % 40 == 0:
        print(f"  {j}/{N} done, elapsed {time.time()-t0:.1f}s", flush=True)

res = {
    "N": N,
    "mean_logit_diff": ld_all.mean().item(),
    "std_logit_diff": ld_all.std().item(),
    "io_over_s_rate": (ld_all > 0).float().mean().item(),
    "mean_io_prob": ioprob_all.mean().item(),
    "duration_seconds": time.time()-t0,
}
print("RESULT", json.dumps(res), flush=True)
with open(os.path.join(OUT, "step2_baseline_result.json"), "w") as f:
    json.dump(res, f, indent=2)
