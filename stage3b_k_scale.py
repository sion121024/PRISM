"""
Stage 3b: K-scale validation — 개선판.

두 가지 테스트:
  (1) 훈련 K 고정 → 추론 K 변화 (이중시계 직접 검증)
      → 에너지 하강이 더 깊어질수록 성능이 올라야 함
  (2) 훈련 K 변화 → 같은 에폭 수 학습 (모델 용량 vs K 트레이드오프)

태스크: char_lm (TinyShakespeare) — assoc_recall보다 어렵고 K-민감도 높음.
"""

import math, time, torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare

BS, BLOCK = 32, 128
D, EMB = 256, 64
TRAIN_K = 4
EPOCHS_SHARED = 3   # 공통 에폭 (빠른 비교용)

train_ds = TinyShakespeare(block_size=BLOCK, split="train")
val_ds   = TinyShakespeare(block_size=BLOCK, split="val")
TL = train_ds.get_loader(batch_size=BS)
VL = val_ds.get_loader(batch_size=BS, shuffle=False)
V  = train_ds.vocab_size

def make_model(K=4, use_gate=False):
    return PRISMLangModel(
        vocab_size=V, d=D, emb_dim=EMB, K=K,
        memory_mode="sliding", mem_rank=8,
        use_urec=True, simple_prior=True,
        use_gate=use_gate,
    )

def train_epochs(model, n_epochs, K_eval=None):
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=n_epochs * len(TL))
    for ep in range(n_epochs):
        model.train()
        for tok in TL:
            o = model(tok)
            opt.zero_grad(); o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

def eval_ppl(model, K_eval=None):
    model.eval()
    vl = 0.0
    k_orig = model.cell.K
    if K_eval is not None:
        model.cell.K = K_eval
    with torch.no_grad():
        for tok in VL:
            vl += model(tok)['loss'].item()
    model.cell.K = k_orig
    return math.exp(vl / len(VL))

print("Stage 3b: K-scale 이중시계 검증")
print("=" * 65)

# ──────────────────────────────────────────────
# 테스트 1: 훈련 K=4 → 추론 K 변화
# ──────────────────────────────────────────────
print("\n[테스트 1] 훈련 K=4 고정, 추론 K 변화")
print(f"  (에폭={EPOCHS_SHARED}, d={D}, gate=False)")
t0 = time.time()
m_train4 = make_model(K=4)
train_epochs(m_train4, EPOCHS_SHARED)

print("  추론 K | val_ppl")
for K_inf in [1, 2, 4, 8, 16]:
    ppl = eval_ppl(m_train4, K_eval=K_inf)
    print(f"     K={K_inf:2d} | {ppl:.2f}")
print(f"  ({time.time()-t0:.0f}s)")

# ──────────────────────────────────────────────
# 테스트 1b: 훈련 K=4+gate → 추론 K 변화
# ──────────────────────────────────────────────
print("\n[테스트 1b] 훈련 K=4+gate, 추론 K 변화")
t0 = time.time()
m_gate4 = make_model(K=4, use_gate=True)
train_epochs(m_gate4, EPOCHS_SHARED)

print("  추론 K | val_ppl")
for K_inf in [1, 2, 4, 8, 16]:
    ppl = eval_ppl(m_gate4, K_eval=K_inf)
    print(f"     K={K_inf:2d} | {ppl:.2f}")
print(f"  ({time.time()-t0:.0f}s)")

# ──────────────────────────────────────────────
# 테스트 2: 훈련 K 변화, 3 에폭
# ──────────────────────────────────────────────
print("\n[테스트 2] 훈련 K 변화 (gate=True 포함), 같은 에폭")
print(f"  (에폭={EPOCHS_SHARED}, d={D})")
for K_tr, gate in [(1, False), (2, False), (4, False), (8, False), (4, True)]:
    t0 = time.time()
    m = make_model(K=K_tr, use_gate=gate)
    train_epochs(m, EPOCHS_SHARED)
    ppl = eval_ppl(m)
    tag = f"K={K_tr}" + ("+gate" if gate else "")
    print(f"  {tag:12s} | {ppl:.2f} ppl ({m.num_params():,} params) [{time.time()-t0:.0f}s]")

print("\n이중시계 가설 검증 요약:")
print("  추론 K↑ → ppl↓ ⇒ 에너지 하강이 추론을 개선함을 지지")
