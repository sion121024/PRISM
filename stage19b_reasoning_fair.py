"""
Stage 19b: 추론 승리 — 공정 파라미터 매칭 (버그 수정판).

Stage 19는 per-token 정규화로 PRISM 기억을 파괴(버그). 폐기.
여기선 Stage 17의 검증된 raw 경로(AssocRecallModel) 그대로 사용하되,
Mamba를 공정 매칭(d48 21.8K vs PRISM 20.3K)으로 비교.

Stage 17은 Mamba-d64(35K, PRISM의 75% 큼)였는데도 n_pairs=12에서 PRISM 승.
공정 매칭이면 PRISM 우위가 더 일찍/크게 나타날 것.
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
VOCAB = KEY_VOCAB + VAL_VOCAB + 1
D, EMB, K = 128, 32, 4
EPOCHS, BS = 25, 128


class MambaRecall(nn.Module):
    def __init__(self, d_model=48):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.mamba = MambaBlock(d_model, d_state=16)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VAL_VOCAB, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.constant_(self.mamba.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, tokens, labels):
        x = self.embed(tokens)
        y, _ = self.mamba(x)
        h = self.norm_f(y[:, -1])
        logits = self.head(h)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        return {'loss': loss, 'acc': acc}

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prism_recall():
    base = PRISMLangModel(vocab_size=VOCAB, d=D, emb_dim=EMB, K=K,
                          memory_mode="sliding", mem_rank=8,
                          use_urec=True, simple_prior=True)
    return AssocRecallModel(base, val_vocab_offset=KEY_VOCAB)


def train_eval(model, loader, vloader):
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
        best = max(best, acc / len(vloader))
    return best


print("Stage 19b: 추론 승리 — 공정 파라미터 매칭")
print(f"  PRISM-d{D} (raw) vs Mamba-d48 | K={K}, epochs={EPOCHS}")
print("=" * 60)

results = {}
for n_pairs in [4, 8, 12, 16]:
    ds = AssocRecallDataset(10000, n_pairs, KEY_VOCAB, VAL_VOCAB)
    vd = AssocRecallDataset(1000, n_pairs, KEY_VOCAB, VAL_VOCAB, seed=99)
    loader, vloader = ds.get_loader(BS), vd.get_loader(BS, shuffle=False)

    t0 = time.time()
    pm = prism_recall(); p_acc = train_eval(pm, loader, vloader)
    p_par = pm.model.num_params()
    mm = MambaRecall(48); m_acc = train_eval(mm, loader, vloader); m_par = mm.num_params()
    win = "PRISM" if p_acc > m_acc else "Mamba"
    results[n_pairs] = (p_acc, p_par, m_acc, m_par, win)
    print(f"n_pairs={n_pairs:2d} | PRISM {p_acc:.3f} ({p_par:,}) | "
          f"Mamba {m_acc:.3f} ({m_par:,}) | 승: {win} | {time.time()-t0:.0f}s")

print("\n" + "=" * 60)
print("요약 (recall accuracy, 공정 파라미터 매칭):")
prism_wins = 0
for n, (pa, pp, ma, mp, w) in results.items():
    print(f"  n_pairs={n:2d}: PRISM {pa:.3f} ({pp/1000:.1f}K) vs Mamba {ma:.3f} ({mp/1000:.1f}K) → {w}")
    if w == "PRISM":
        prism_wins += 1
print(f"\nPRISM 승: {prism_wins}/{len(results)} 난이도")
