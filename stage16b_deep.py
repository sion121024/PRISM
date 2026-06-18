"""
Stage 16b: 더 깊은 사고 + 더 큰 기억으로 Mamba 격파 시도.

Stage 16 (gate+conv+all, K=4, rank16) → ep6 9.10, 수렴 ~6.5-7 예상.
conv 무용·MLP≈linear 판명 → conv 제거, 대신 설계철학 정통 레버 강화:
  - K=6 (더 깊은 에너지 하강 = 더 깊은 사고)
  - mem_rank=24 (더 큰 Hebbian 연상기억)
  - gate + input_dep_pi + momentum0.9 + prior_bias

목표: Mamba-d128 6.45 ppl 격파.
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare

EPOCHS, BS, BLOCK = 20, 32, 128
D, EMB, K = 256, 64, 6

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

model = PRISMLangModel(
    vocab_size=V, d=D, emb_dim=EMB, K=K,
    memory_mode="sliding", mem_rank=24,
    use_urec=True, simple_prior=True,
    use_gate=True,
    input_dep_pi=True, momentum=0.9, prior_bias=True,
)
print(f"PRISM gate+sel+mom+prior (K={K}, rank=24) | {model.num_params():,} params")
print(f"  목표: Mamba-d128 6.45 ppl 격파 | {EPOCHS} epochs")
print("=" * 60)

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
    vl /= len(VL); ppl = math.exp(vl)
    if ppl < best:
        best = ppl
        torch.save(model.state_dict(), "best_prism_deep.pt")
    flag = "  ← Mamba 격파!" if ppl < 6.45 else ""
    print(f"  ep{ep:2d} | train={tl/len(TL):.4f} | val_ppl={ppl:.2f} | {time.time()-t0:.0f}s{flag}")

print("=" * 60)
print(f"PRISM(deep) 수렴점: {best:.2f} ppl  vs  Mamba-d128: 6.45 ppl")
print("승리!" if best < 6.45 else f"격차: +{best-6.45:.2f} ppl")
