"""
v1 vs v2 실제 학습 비교 테스트.
같은 데이터, 같은 하이퍼파라미터로 10 epoch 훈련 후 ppl 비교.
"""

import math, time, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from prism import PRISMLM, CharDataset
from prism.model_v2 import PRISMBase

TEXT = open("/tmp/mini_shake.txt").read()
DS   = CharDataset(TEXT, seq_len=64)
n_val = max(1, len(DS)//10)
tr, va = random_split(DS, [len(DS)-n_val, n_val], generator=torch.Generator().manual_seed(42))
TL = DataLoader(tr, batch_size=64, shuffle=True,  drop_last=True)
VL = DataLoader(va, batch_size=64, shuffle=False)

EPOCHS = 10
D      = 128
VOCAB  = DS.vocab_size

@torch.no_grad()
def ppl(model, K=None, is_v2=False):
    model.eval()
    tot, n = 0.0, 0
    for x, y in VL:
        logits = model(x, K=K) if is_v2 else model(x, K=K, use_deq=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        tot += loss.item()*x.numel(); n += x.numel()
    return math.exp(tot/n)

def train(model, is_v2=False, K_train=None):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(TL))
    log = []
    for ep in range(1, EPOCHS+1):
        model.train()
        for x, y in TL:
            logits = model(x, K=K_train) if is_v2 else model(x, K=K_train, use_deq=False)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        p = ppl(model, is_v2=is_v2)
        log.append(p)
        print(f"  ep{ep:2d}  ppl={p:.2f}")
    return log

print("="*55)
print(f"v1  PRISMLM    d={D} K=8  params={sum(p.numel() for p in PRISMLM(VOCAB,d=D).parameters()):,}")
t0 = time.time()
v1 = PRISMLM(VOCAB, d=D, K=8)
log_v1 = train(v1, is_v2=False, K_train=8)
t_v1 = time.time()-t0

print()
print("="*55)
print(f"v2  PRISMBase  d={D} K=6  params={sum(p.numel() for p in PRISMBase(VOCAB,d=D,K=6).parameters()):,}")
t0 = time.time()
v2 = PRISMBase(VOCAB, d=D, K=6)
log_v2 = train(v2, is_v2=True, K_train=6)
t_v2 = time.time()-t0

print()
print("="*55)
print("결과 요약")
print("="*55)
print(f"v1  best ppl={min(log_v1):.2f}  time={t_v1:.0f}s")
print(f"v2  best ppl={min(log_v2):.2f}  time={t_v2:.0f}s")
print()

# 더 생각하면 더 잘 되나? (K 늘려서 eval)
p_v2_k24 = ppl(v2, K=24, is_v2=True)
p_v1_k24 = ppl(v1, K=24, is_v2=False)
print(f"v1  K=24 (no retrain)  ppl={p_v1_k24:.2f}")
print(f"v2  K=24 (no retrain)  ppl={p_v2_k24:.2f}")
print(f"  {'v2 gains more from deeper thinking ✓' if (min(log_v2)-p_v2_k24) > (min(log_v1)-p_v1_k24) else 'v1 gains more'}")
