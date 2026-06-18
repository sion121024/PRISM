"""
Stage 15: 비볼록 에너지 (MLP decoder) — 설계철학 가장 순수한 레버.

linear decoder: E(x) = ½‖u−Dx‖² → 2차식, 사고가 자명 (closed-form 최솟값).
MLP decoder:    E(x) = ½‖u−W2·tanh(W1 x)‖² → 비볼록, K-step이 진짜 추론.

"진짜 사고는 비볼록 에너지에서만 일어난다" — K-effect가 MLP에서 더 커야 함.

비교 (gate+conv 고정, decoder만 변화):
  A) linear  (96K)
  B) mlp64   (100K)  ← param-matched
  C) mlp128  (121K)
같은 5 epoch, char_lm.
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare

EPOCHS, BS, BLOCK = 5, 32, 128
D, EMB, K = 256, 64, 4

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

def prism(**kw):
    return PRISMLangModel(vocab_size=V, d=D, emb_dim=EMB, K=K,
                          memory_mode="sliding", mem_rank=8,
                          use_urec=True, simple_prior=True,
                          use_gate=True, use_conv=True, **kw)

def run(name, model):
    p = model.num_params()
    print(f"\n--- {name} | {p:,} params ---")
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS * len(TL))
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl = 0.0
        for tok in TL:
            o = model(tok); opt.zero_grad(); o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += o['loss'].item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for tok in VL:
                vl += model(tok)['loss'].item()
        vl /= len(VL); ppl = math.exp(vl); best = min(best, ppl)
        print(f"  ep{ep} | train={tl/len(TL):.4f} | val_ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    print(f"  Best: {best:.2f}")
    return best

print("Stage 15: 비볼록 에너지 (MLP decoder)")
print("=" * 60)
res = {}
res["A) linear"] = run("A) linear",  prism())
res["B) mlp64"]  = run("B) mlp64",   prism(decoder="mlp", dec_hidden=64))
res["C) mlp128"] = run("C) mlp128",  prism(decoder="mlp", dec_hidden=128))
print("\n" + "=" * 60)
print("요약:")
for k, v in res.items():
    print(f"  {k}: {v:.2f} ppl")
