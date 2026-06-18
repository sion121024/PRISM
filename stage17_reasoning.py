"""
Stage 17: 추론 task에서 PRISM vs Mamba — 이중시계가 빛나는 전장.

char-LM은 빠른 n-gram 포착(Mamba 강점)을 보상 → PRISM에 불리.
연상 회상(associative recall)은 명시적 content-addressable 기억 + 반복 정제 필요
→ PRISM의 Hebbian 기억 M + K-step 에너지 하강이 설계상 유리.

가설:
  (1) 난이도↑(n_pairs↑)에서 PRISM이 Mamba보다 우아하게 저하
  (2) 어려운 인스턴스일수록 K↑가 더 도움 (이중시계)

파라미터 매칭, 동일 학습 설정.
"""

import math, time, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import AssocRecallDataset
from tasks.assoc_recall import AssocRecallModel
from baselines.mamba_lm import MambaBlock

KEY_VOCAB = VAL_VOCAB = 16
D, EMB, K = 128, 32, 4
EPOCHS, BS = 25, 128


class MambaRecallModel(nn.Module):
    """Mamba 연상기억 래퍼 — 전체 시퀀스 처리 후 마지막 위치로 value 예측."""
    def __init__(self, vocab_size, d_model, val_vocab_offset, val_vocab, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, val_vocab, bias=False)
        self.offset = val_vocab_offset
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.constant_(self.mamba.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, tokens, labels):
        x = self.embed(tokens)               # [B, T, d]
        y, _ = self.mamba(x)                 # [B, T, d]
        h = self.norm_f(y[:, -1])            # 마지막 위치
        logits = self.head(h)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        return {'loss': loss, 'acc': acc}

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_loaders(n_pairs):
    ds = AssocRecallDataset(10000, n_pairs, KEY_VOCAB, VAL_VOCAB)
    vd = AssocRecallDataset(1000, n_pairs, KEY_VOCAB, VAL_VOCAB, seed=99)
    return ds, ds.get_loader(BS), vd.get_loader(BS, shuffle=False)


def train_eval(model, loader, vloader, is_prism):
    params = model.num_params() if hasattr(model, 'num_params') else \
             sum(p.numel() for p in model.parameters())
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS * len(loader))
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for tok, lab in loader:
            o = model(tok, lab)
            opt.zero_grad(); o['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        model.eval(); acc = 0.0
        with torch.no_grad():
            for tok, lab in vloader:
                acc += model(tok, lab)['acc']
        acc /= len(vloader)
        best = max(best, acc)
    return best, params


def prism_recall(vocab):
    base = PRISMLangModel(vocab_size=vocab, d=D, emb_dim=EMB, K=K,
                          memory_mode="sliding", mem_rank=8,
                          use_urec=True, simple_prior=True)
    return AssocRecallModel(base, val_vocab_offset=KEY_VOCAB), base


print("Stage 17: 추론 task — PRISM vs Mamba (연상 회상)")
print(f"  d={D}, K={K}, epochs={EPOCHS}")
print("=" * 60)

results = {}
for n_pairs in [4, 8, 12]:
    vocab = KEY_VOCAB + VAL_VOCAB + 1
    ds, loader, vloader = make_loaders(n_pairs)

    t0 = time.time()
    pm, base = prism_recall(vocab)
    p_acc, p_par = train_eval(pm, loader, vloader, True)

    mm = MambaRecallModel(vocab, d_model=64, val_vocab_offset=KEY_VOCAB,
                          val_vocab=VAL_VOCAB)
    m_acc, m_par = train_eval(mm, loader, vloader, False)

    winner = "PRISM" if p_acc > m_acc else "Mamba"
    results[n_pairs] = (p_acc, p_par, m_acc, m_par, winner)
    print(f"n_pairs={n_pairs:2d} | PRISM {p_acc:.3f} ({p_par:,}) | "
          f"Mamba {m_acc:.3f} ({m_par:,}) | 승: {winner} | {time.time()-t0:.0f}s")

print("\n" + "=" * 60)
print("추론 task 요약 (recall accuracy, 높을수록 좋음):")
for n, (pa, pp, ma, mp, w) in results.items():
    print(f"  n_pairs={n:2d}: PRISM {pa:.3f} vs Mamba {ma:.3f} → {w}")

# 이중시계: 가장 어려운 난이도에서 PRISM K 스윕
print("\n[이중시계] n_pairs=12, 훈련 K=4 모델, 추론 K 변화:")
vocab = KEY_VOCAB + VAL_VOCAB + 1
ds, loader, vloader = make_loaders(12)
pm, base = prism_recall(vocab)
train_eval(pm, loader, vloader, True)  # 학습
for K_inf in [1, 2, 4, 8, 16]:
    base.cell.K = K_inf
    pm.eval(); acc = 0.0
    with torch.no_grad():
        for tok, lab in vloader:
            acc += pm(tok, lab)['acc']
    acc /= len(vloader)
    print(f"  추론 K={K_inf:2d} | acc={acc:.3f}")
