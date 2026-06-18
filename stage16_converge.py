"""
Stage 16: 수렴점 비교 — PRISM이 충분한 epoch에서 Mamba를 이기는가?

Mamba는 ep3-4에 수렴 (d96=6.82, d128=6.45).
PRISM gate는 ep5에도 미수렴 (13.61, 계속 하강) — Hebbian 기억 충전에 시간 필요.

공정 비교: PRISM 최강 fast 설정을 20 epoch까지 → 수렴점 vs Mamba 6.45.
설계철학 준수 설정만 사용 (gate=readout-only, conv, input_dep_pi, momentum, prior_bias).
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare

EPOCHS, BS, BLOCK = 20, 32, 128
D, EMB, K = 256, 64, 4

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

model = PRISMLangModel(
    vocab_size=V, d=D, emb_dim=EMB, K=K,
    memory_mode="sliding", mem_rank=16,
    use_urec=True, simple_prior=True,
    use_gate=True, use_conv=True,
    input_dep_pi=True, momentum=0.9, prior_bias=True,
)
print(f"PRISM gate+conv+all (수렴 테스트) | {model.num_params():,} params")
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
        torch.save(model.state_dict(), "best_prism_converge.pt")
    flag = "  ← Mamba 격파!" if ppl < 6.45 else ""
    print(f"  ep{ep:2d} | train={tl/len(TL):.4f} | val_ppl={ppl:.2f} | {time.time()-t0:.0f}s{flag}")

print("=" * 60)
print(f"PRISM 수렴점: {best:.2f} ppl  vs  Mamba-d128: 6.45 ppl")
print("승리" if best < 6.45 else f"격차: +{best-6.45:.2f} ppl")
