"""
Stage 14c: 게이트 효과 체계적 검증 + Stage 14b 보완.

use_gate(readout-only)가 지금까지 단일 최대 개선인 것을 확인:
Stage 14 multilayer에서 13.14 ppl (baseline 24.18 대비 1.84×)

비교:
  A) baseline         : simple_prior + urec, K=4
  B) +gate            : baseline + readout gate (13.14 ← 이미 측정)
  C) +gate+conv       : B + depthwise conv1d
  D) +gate+conv+all   : C + input_dep_pi + momentum + prior_bias
  E) +gate+2layers    : B + hierarchical (10 epochs for proper convergence)
  F) Mamba (ref)      : 파라미터 유사 비교기준

5 epochs (E는 10 epochs).
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare
from baselines import MambaLangModel  # local Mamba proxy

EPOCHS  = 5
BS      = 32
BLOCK   = 128
D, EMB  = 256, 64
K       = 4

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

def prism(**kw):
    return PRISMLangModel(
        vocab_size=V, d=D, emb_dim=EMB, K=K,
        memory_mode="sliding", mem_rank=8,
        use_urec=True, simple_prior=True,
        **kw,
    )

def run(name, model, epochs=EPOCHS):
    p = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"\n--- {name} | {p:,} params ---")
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs * len(TL))
    best = float('inf')
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tl = 0.0
        for tok in TL:
            o = model(tok)
            opt.zero_grad(); o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += o['loss'].item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for tok in VL:
                vl += model(tok)['loss'].item()
        vl /= len(VL)
        ppl = math.exp(vl)
        if ppl < best:
            best = ppl
        print(f"  ep{ep} | train={tl/len(TL):.4f} | val_ppl={ppl:.2f} | {time.time()-t0:.1f}s")
    print(f"  Best: {best:.2f}")
    return best

configs = [
    ("A) baseline",          prism()),
    ("B) gate",              prism(use_gate=True)),
    ("C) gate+conv",         prism(use_gate=True, use_conv=True)),
    ("D) gate+conv+all",     prism(use_gate=True, use_conv=True,
                                   input_dep_pi=True, momentum=0.9, prior_bias=True)),
]

print("Stage 14c: 게이트 효과 체계적 검증")
print(f"  d={D}, emb={EMB}, K={K}, block={BLOCK}, epochs={EPOCHS}")
print("=" * 65)

results = {}
for name, model in configs:
    results[name] = run(name, model)

# 2-layer는 더 길게
print("\n--- E) gate+2layers (10 epochs for convergence) ---")
m2 = prism(use_gate=True, n_layers=2)
results["E) gate+2layers"] = run("E) gate+2layers", m2, epochs=10)

# Mamba reference
try:
    mamba = MambaLangModel(vocab_size=V, d_model=256, d_state=16, n_layers=2, emb_dim=EMB)
    results["F) Mamba"] = run("F) Mamba", mamba)
except Exception as e:
    print(f"\nMamba skipped: {e}")

print()
print("=" * 65)
print("최종 요약 (val ppl, 낮을수록 좋음):")
for name, ppl in results.items():
    print(f"  {name}: {ppl:.2f}")
