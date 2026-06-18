"""
Mamba 기준선 — PRISM과 파라미터 매칭, 동일 학습 설정.

PRISM gate+conv+all (117K) vs Mamba d128 (130K) head-to-head.
같은 데이터/에폭/block_size로 공정 비교.
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from baselines import MambaLangModel
from tasks import TinyShakespeare

EPOCHS, BS, BLOCK = 5, 32, 128

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

def run(name, model):
    p = sum(x.numel() for x in model.parameters())
    print(f"\n--- {name} | {p:,} params ---")
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS * len(TL))
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); tl = 0.0
        for tok in TL:
            o = model(tok)
            loss = o['loss'] if isinstance(o, dict) else o
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += loss.item()
        model.eval(); vl = 0.0
        with torch.no_grad():
            for tok in VL:
                o = model(tok)
                vl += (o['loss'] if isinstance(o, dict) else o).item()
        vl /= len(VL); ppl = math.exp(vl)
        best = min(best, ppl)
        print(f"  ep{ep} | train={tl/len(TL):.4f} | val_ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    print(f"  Best: {best:.2f}")
    return best

print("Mamba 기준선 (PRISM 파라미터 매칭)")
print(f"  block={BLOCK}, epochs={EPOCHS}, B={BS}")
print("=" * 60)

res = {}
res["Mamba-d96 (79K, ~PRISM baseline)"]  = run("Mamba-d96",  MambaLangModel(vocab_size=V, d_model=96,  d_state=16))
res["Mamba-d128 (130K, ~PRISM gate+conv+all)"] = run("Mamba-d128", MambaLangModel(vocab_size=V, d_model=128, d_state=16))

print("\n" + "=" * 60)
print("Mamba 기준선 요약:")
for k, v in res.items():
    print(f"  {k}: {v:.2f} ppl")
