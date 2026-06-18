"""
Stage 19: 추론 task 승리 견고화 — urec 적용 + 공정 파라미터 매칭.

Stage 17은 raw 경로(urec 없음)로도 n_pairs=12에서 PRISM 승.
여기선:
  (1) urec(설계상 필수) + gate readout 적용한 완전한 PRISM 파이프라인
  (2) 공정 파라미터 매칭: Mamba-d48 (21.8K) vs PRISM-d128 (24.5K)
  (3) 난이도 확장: n_pairs = 4, 8, 12, 16

가설: 완전한 PRISM 파이프라인 + 공정 매칭에서 PRISM 우위가 더 일찍·크게 나타남.
"""

import math, time, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import AssocRecallDataset
from baselines.mamba_lm import MambaBlock

KEY_VOCAB = VAL_VOCAB = 16
VOCAB = KEY_VOCAB + VAL_VOCAB + 1
D, EMB, K = 128, 32, 4
EPOCHS, BS = 25, 128


class PRISMRecall(nn.Module):
    """완전한 PRISM 파이프라인 recall: urec → 에너지 하강 → gate readout → value."""
    def __init__(self):
        super().__init__()
        self.m = PRISMLangModel(vocab_size=VOCAB, d=D, emb_dim=EMB, K=K,
                                memory_mode="sliding", mem_rank=8,
                                use_urec=True, simple_prior=True, use_gate=True)

    def forward(self, tokens, labels):
        m = self.m
        B, T = tokens.shape
        x, mem = m.init_state(B, tokens.device)
        u_last_raw = None
        for t in range(T):
            u_raw = m.embed(tokens[:, t])
            u = m.u_rec2(F.gelu(m.u_rec1(torch.cat([u_raw, x], dim=-1))))
            x_prior = x  # simple_prior
            x0 = None if t == 0 else x
            x, mem = m.cell(u, mem, x0, training=self.training, x_prior=x_prior)
            x = x * x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            u_last_raw = u_raw
        h = m.norm_f(x)
        h = h * F.silu(m.gate_proj(u_last_raw))   # readout gate
        logits = m.output_proj(h)
        logits_val = logits[:, KEY_VOCAB:KEY_VOCAB + VAL_VOCAB]
        loss = F.cross_entropy(logits_val, labels)
        acc = (logits_val.argmax(-1) == labels).float().mean().item()
        return {'loss': loss, 'acc': acc}

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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


print("Stage 19: 추론 승리 견고화 (urec + 공정 매칭)")
print(f"  PRISM-d{D} vs Mamba-d48 | K={K}, epochs={EPOCHS}")
print("=" * 60)

results = {}
for n_pairs in [4, 8, 12, 16]:
    ds = AssocRecallDataset(10000, n_pairs, KEY_VOCAB, VAL_VOCAB)
    vd = AssocRecallDataset(1000, n_pairs, KEY_VOCAB, VAL_VOCAB, seed=99)
    loader, vloader = ds.get_loader(BS), vd.get_loader(BS, shuffle=False)

    t0 = time.time()
    pm = PRISMRecall(); p_acc = train_eval(pm, loader, vloader); p_par = pm.num_params()
    mm = MambaRecall(48); m_acc = train_eval(mm, loader, vloader); m_par = mm.num_params()
    win = "PRISM" if p_acc > m_acc else "Mamba"
    results[n_pairs] = (p_acc, p_par, m_acc, m_par, win)
    print(f"n_pairs={n_pairs:2d} | PRISM {p_acc:.3f} ({p_par:,}) | "
          f"Mamba {m_acc:.3f} ({m_par:,}) | 승: {win} | {time.time()-t0:.0f}s")

print("\n" + "=" * 60)
print("요약 (recall accuracy):")
prism_wins = 0
for n, (pa, pp, ma, mp, w) in results.items():
    print(f"  n_pairs={n:2d}: PRISM {pa:.3f} vs Mamba {ma:.3f} → {w}")
    if w == "PRISM":
        prism_wins += 1
print(f"\nPRISM 승: {prism_wins}/{len(results)} 난이도")
