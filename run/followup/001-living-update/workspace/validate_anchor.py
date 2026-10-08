"""Validate faithfulness harness on GPT-2 small with the PAPER circuit.
Expect F(M)~3.26, F(C)~2.87, F(C)/F(M)~0.878 (replication step8)."""
import torch
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.ioi_dataset import IOIDataset
from easy_transformer.ioi_utils import logit_diff
from easy_transformer.ioi_circuit_extraction import (
    CIRCUIT, get_heads_circuit, do_circuit_extraction, get_extracted_idx)

torch.set_grad_enabled(False)
m = EasyTransformer.from_pretrained("gpt2").cuda()
m.set_use_attn_result(True)
d = IOIDataset(prompt_type="mixed", N=100, tokenizer=m.tokenizer)  # step8 used default prepend_bos
abc = (d.gen_flipped_prompts(("IO","RAND")).gen_flipped_prompts(("S","RAND")).gen_flipped_prompts(("S1","RAND")))

# path 1: replication step8 exact (get_heads_circuit uses global RELEVANT_TOKENS)
m.reset_hooks(); F_M = logit_diff(m, d).item()
htk = get_heads_circuit(d, excluded=[], circuit=CIRCUIT.copy())
m.reset_hooks()
m2,_ = do_circuit_extraction(model=m, heads_to_keep=htk, mlps_to_remove={}, ioi_dataset=d, mean_dataset=abc)
F_C = logit_diff(m2, d).item(); m.reset_hooks()
print(f"[step8-exact]     F(M)={F_M:.4f} F(C)={F_C:.4f} ratio={F_C/F_M:.4f}")

# path 2: my manual heads_to_keep (as in ladder_tier1) using paper circuit
CLASS_TOKENS = {"name mover":["end"],"negative":["end"],"s2 inhibition":["end"],
                "induction":["S2"],"duplicate token":["S2"],"previous token":["S+1"]}
htk2={}
for cls,heads in CIRCUIT.items():
    for h in heads:
        htk2[tuple(h)] = get_extracted_idx(CLASS_TOKENS[cls], d)
m.reset_hooks()
m3,_ = do_circuit_extraction(model=m, heads_to_keep=htk2, mlps_to_remove={}, ioi_dataset=d, mean_dataset=abc)
F_C2 = logit_diff(m3, d).item(); m.reset_hooks()
print(f"[my-manual-paper] F(M)={F_M:.4f} F(C)={F_C2:.4f} ratio={F_C2/F_M:.4f}")
