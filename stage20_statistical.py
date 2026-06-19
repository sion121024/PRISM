"""
Stage 20: 통계적 검증 — 다중 시드, 추가 베이스라인, 유의성 검정

핵심 주장: "n_pairs=12에서 PRISM이 Mamba보다 유의미하게 높은 recall 달성"

검증 방법:
  - n_pairs=12 (PRISM 승리 주장 핵심 케이스)
  - 4개 모델: PRISM, Mamba-d48, GRU-d52, Transformer-d40
  - 3 seeds (0, 1, 2) → mean ± std
  - Welch's t-test: PRISM vs 각 베이스라인
  - epochs=15, BS=128, LR=3e-4

실험 환경:
  CPU x86_64 (4코어), RAM 15GB
  PyTorch 2.12, Python 3.11
  torch.set_num_threads(1)

재현:
  pip install torch scipy
  python stage20_statistical.py
"""

import math, time, torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats as scipy_stats

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks.assoc_recall import AssocRecallModel, AssocRecallDataset
from baselines.mamba_lm import MambaBlock

KEY_VOCAB = VAL_VOCAB = 16
VOCAB = KEY_VOCAB + VAL_VOCAB + 1
D, EMB, K = 128, 32, 4
EPOCHS, BS = 15, 128
SEEDS = [0, 1, 2]
LR, WD = 3e-4, 1e-4
N_PAIRS = 12   # 핵심 케이스


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
        return {'loss': loss, 'acc': (logits.argmax(-1) == labels).float().mean().item()}

    def num_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GRURecall(nn.Module):
    def __init__(self, d_model=52):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VAL_VOCAB, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens, labels):
        x = self.embed(tokens)
        y, _ = self.gru(x)
        h = self.norm_f(y[:, -1])
        logits = self.head(h)
        loss = F.cross_entropy(logits, labels)
        return {'loss': loss, 'acc': (logits.argmax(-1) == labels).float().mean().item()}

    def num_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerRecall(nn.Module):
    def __init__(self, d_model=40, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d_model)
        self.pos_enc = nn.Embedding(256, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VAL_VOCAB, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, tokens, labels):
        T = tokens.size(1)
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_enc(pos)
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        y = self.transformer(x, mask=mask, is_causal=True)
        h = self.norm_f(y[:, -1])
        logits = self.head(h)
        loss = F.cross_entropy(logits, labels)
        return {'loss': loss, 'acc': (logits.argmax(-1) == labels).float().mean().item()}

    def num_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prism_factory():
    base = PRISMLangModel(vocab_size=VOCAB, d=D, emb_dim=EMB, K=K,
                          memory_mode="sliding", mem_rank=8,
                          use_urec=True, simple_prior=True)
    return AssocRecallModel(base, val_vocab_offset=KEY_VOCAB)


def train_eval(model, loader, vloader, seed):
    torch.manual_seed(seed)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
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


print("Stage 20: 통계적 검증 (n_pairs=12, 3 seeds)")
print(f"  epochs={EPOCHS}, BS={BS}, LR={LR}, K={K}")
print(f"  seeds={SEEDS}  |  torch={torch.__version__}")
print("=" * 70)

MODEL_CONFIGS = [
    ("PRISM",        prism_factory),
    ("Mamba-d48",    lambda: MambaRecall(48)),
    ("GRU-d52",      lambda: GRURecall(52)),
    ("Transformer",  lambda: TransformerRecall(40, n_heads=4, n_layers=2)),
]

print(f"\n[n_pairs={N_PAIRS}]")
print(f"{'모델':15s} {'params':>8} | {'seed0':>7} {'seed1':>7} {'seed2':>7} | {'mean±std':>14}")
print("-" * 72)

all_results = {}
t_start = time.time()

for name, factory in MODEL_CONFIGS:
    accs = []
    t0 = time.time()
    ds = [AssocRecallDataset(10000, N_PAIRS, KEY_VOCAB, VAL_VOCAB, seed=s) for s in SEEDS]
    vd = [AssocRecallDataset(1000,  N_PAIRS, KEY_VOCAB, VAL_VOCAB, seed=100+s) for s in SEEDS]

    n_params = None
    for i, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        model = factory()
        if n_params is None:
            n_params = (model.model.num_params() if hasattr(model, 'model')
                        else model.num_params())
        loader  = ds[i].get_loader(BS)
        vloader = vd[i].get_loader(BS, shuffle=False)
        acc = train_eval(model, loader, vloader, seed)
        accs.append(acc)

    mean = sum(accs) / len(accs)
    std  = (sum((a - mean)**2 for a in accs) / (len(accs) - 1)) ** 0.5
    all_results[name] = (accs, n_params)
    print(f"  {name:13s} {n_params:>8,} | "
          + "  ".join(f"{a:.3f}" for a in accs)
          + f"  | {mean:.3f} ± {std:.3f}  ({time.time()-t0:.0f}s)")

# ── 통계 검정 ──
print("\n" + "=" * 70)
print("Welch's t-test (PRISM vs 각 베이스라인)")
print("-" * 70)
prism_accs, _ = all_results["PRISM"]
for name, _ in MODEL_CONFIGS[1:]:
    other_accs, _ = all_results[name]
    t_stat, p_val = scipy_stats.ttest_ind(prism_accs, other_accs, equal_var=False)
    prism_m = sum(prism_accs) / len(prism_accs)
    other_m = sum(other_accs) / len(other_accs)
    diff = prism_m - other_m
    winner = "PRISM" if diff > 0 else name
    sig = ("p<0.05 ✅" if p_val < 0.05
           else "p<0.10 ⚠ " if p_val < 0.10
           else "비유의 ❌")
    print(f"  PRISM vs {name:13s}: Δ={diff:+.3f}  t={t_stat:+.2f}  p={p_val:.4f}  {sig}  → {winner} 우세")

# ── 전체 순위 ──
print("\n" + "=" * 70)
print("n_pairs=12 최종 순위 (mean recall accuracy)")
ranked = sorted([(sum(v[0])/len(v[0]), name) for name, v in all_results.items()], reverse=True)
for rank, (mean, name) in enumerate(ranked, 1):
    accs, p = all_results[name]
    std = (sum((a - mean)**2 for a in accs) / (len(accs) - 1)) ** 0.5
    print(f"  {rank}위  {name:13s}: {mean:.3f} ± {std:.3f}  ({p:,} params)")

print(f"\n총 실험 시간: {time.time()-t_start:.0f}s")
print("\n주의: n=3 seeds는 최소 기준. GPU 환경에서 5+ seeds 권장.")
