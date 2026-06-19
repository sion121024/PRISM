"""
Stage 4 (추론 축): GPU 스케일업 — 연상 회상 head-to-head + 이중시계.

PRISM의 입증된 전장(부하 하 추론)을 GPU로 확장:
  - CPU(d=128, 20K) → GPU(d=256~512) 대형화
  - 더 어려운 난이도(n_pairs ↑, vocab ↑)에서 PRISM의 명시적 Hebbian 기억이
    Mamba의 고정크기 압축 상태보다 강건한지 재확인
  - 이중시계: 같은 모델에서 추론 K ↑ → 어려운 인스턴스 정확도 ↑ (단조)

설계철학 유지: stage19b의 검증된 raw 경로(AssocRecallModel) 사용,
공정 파라미터 매칭(Mamba를 PRISM에 맞춤).

실행:
  python stage21_reasoning_scaleup.py --smoke
  python stage21_reasoning_scaleup.py --device cuda --d 384
"""

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from prism import PRISMLangModel
from tasks import AssocRecallDataset
from tasks.assoc_recall import AssocRecallModel
from baselines.mamba_lm import MambaBlock


class MambaRecall(nn.Module):
    def __init__(self, vocab, val_vocab, d_model, d_state=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state)
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, val_vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.constant_(self.mamba.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, tokens, labels):
        x = self.embed(tokens)
        y, _ = self.mamba(x)
        logits = self.head(self.norm_f(y[:, -1]))
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(-1) == labels).float().mean().item()
        return {"loss": loss, "acc": acc}

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def prism_recall(vocab, key_vocab, d, emb, K, mem_rank):
    base = PRISMLangModel(vocab_size=vocab, d=d, emb_dim=emb, K=K,
                          memory_mode="sliding", mem_rank=mem_rank,
                          mem_scale=4.0, use_urec=True, simple_prior=True)
    return AssocRecallModel(base, val_vocab_offset=key_vocab)


def match_mamba_d(target_params, vocab, val_vocab, lo=32, hi=512):
    """PRISM params에 가장 가까운 Mamba d_model 탐색."""
    best, bestd = None, lo
    for d in range(lo, hi + 1, 8):
        n = MambaRecall(vocab, val_vocab, d).num_params()
        if best is None or abs(n - target_params) < best:
            best, bestd = abs(n - target_params), d
    return bestd


def train_eval(model, loader, vloader, epochs, lr, device, is_prism):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=epochs * len(loader))
    best = 0.0
    for ep in range(epochs):
        model.train()
        for tok, lab in loader:
            tok, lab = tok.to(device), lab.to(device)
            o = model(tok, lab)
            opt.zero_grad(); o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
        model.eval(); acc = 0.0
        with torch.no_grad():
            for tok, lab in vloader:
                tok, lab = tok.to(device), lab.to(device)
                acc += model(tok, lab)["acc"]
        best = max(best, acc / len(vloader))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--mem_rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--key_vocab", type=int, default=24)
    p.add_argument("--val_vocab", type=int, default=24)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--pairs", type=str, default="8,12,16,20")
    p.add_argument("--k_sweep", type=str, default="1,2,4,8",
                   help="이중시계 검증용 추론 K (가장 어려운 난이도에서)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.device = "cpu"; args.d = 64; args.emb = 32; args.K = 2
        args.mem_rank = 8; args.epochs = 2; args.batch_size = 64
        args.key_vocab = args.val_vocab = 8
        args.n_train = 1000; args.n_val = 200
        args.pairs = "4,6"; args.k_sweep = "1,2"

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cpu":
        torch.set_num_threads(4)
    device = torch.device(args.device)
    vocab = args.key_vocab + args.val_vocab + 1
    pairs = [int(x) for x in args.pairs.split(",")]
    k_sweep = [int(x) for x in args.k_sweep.split(",")]

    print("=" * 64)
    print("Stage 4 (추론): GPU 스케일업 — 연상 회상 + 이중시계")
    print("=" * 64)
    print(f"device={device} | PRISM d={args.d} K={args.K} rank={args.mem_rank} | "
          f"key/val_vocab={args.key_vocab}/{args.val_vocab} | epochs={args.epochs}")

    # ---- 난이도 sweep (공정 매칭) ---- #
    results = {}
    prism_par = mamba_par = mamba_d = None
    for npairs in pairs:
        ds = AssocRecallDataset(args.n_train, npairs, args.key_vocab, args.val_vocab)
        vd = AssocRecallDataset(args.n_val, npairs, args.key_vocab, args.val_vocab, seed=99)
        loader = ds.get_loader(args.batch_size)
        vloader = vd.get_loader(args.batch_size, shuffle=False)

        t0 = time.time()
        pm = prism_recall(vocab, args.key_vocab, args.d, args.emb, args.K, args.mem_rank)
        prism_par = pm.model.num_params()
        if mamba_d is None:
            mamba_d = match_mamba_d(prism_par, vocab, args.val_vocab)
        p_acc = train_eval(pm, loader, vloader, args.epochs, args.lr, device, True)

        mm = MambaRecall(vocab, args.val_vocab, mamba_d)
        mamba_par = mm.num_params()
        m_acc = train_eval(mm, loader, vloader, args.epochs, args.lr, device, False)

        win = "PRISM" if p_acc > m_acc else "Mamba"
        results[npairs] = (p_acc, m_acc, win)
        print(f"n_pairs={npairs:2d} | PRISM {p_acc:.3f} ({prism_par/1000:.0f}K) | "
              f"Mamba {m_acc:.3f} ({mamba_par/1000:.0f}K) | 승:{win} | {time.time()-t0:.0f}s",
              flush=True)

    # ---- 이중시계: 가장 어려운 난이도에서 추론 K sweep ---- #
    hard = max(pairs)
    print(f"\n이중시계 검증 (n_pairs={hard}, 학습 K={args.K}, 추론 K 변경):")
    ds = AssocRecallDataset(args.n_train, hard, args.key_vocab, args.val_vocab)
    vd = AssocRecallDataset(args.n_val, hard, args.key_vocab, args.val_vocab, seed=99)
    loader = ds.get_loader(args.batch_size)
    vloader = vd.get_loader(args.batch_size, shuffle=False)
    pm = prism_recall(vocab, args.key_vocab, args.d, args.emb, args.K, args.mem_rank)
    train_eval(pm, loader, vloader, args.epochs, args.lr, device, True)
    pm.eval()
    k_results = {}
    for Kinf in k_sweep:
        pm.model.cell.K = Kinf
        acc = 0.0
        with torch.no_grad():
            for tok, lab in vloader:
                tok, lab = tok.to(device), lab.to(device)
                acc += pm(tok, lab)["acc"]
        k_results[Kinf] = acc / len(vloader)
        print(f"  추론 K={Kinf:2d} → acc {k_results[Kinf]:.3f}", flush=True)
    mono = all(k_results[k_sweep[i]] <= k_results[k_sweep[i+1]] + 1e-3
               for i in range(len(k_sweep) - 1))
    print(f"  이중시계 단조성(K↑→acc↑): {'✓' if mono else '대체로 ✓ 아님'}")

    print("\n" + "=" * 64)
    print(f"요약 (공정 매칭: PRISM {prism_par/1000:.0f}K vs Mamba-d{mamba_d} {mamba_par/1000:.0f}K):")
    pw = sum(1 for _, _, w in results.values() if w == "PRISM")
    for n, (pa, ma, w) in results.items():
        print(f"  n_pairs={n:2d}: PRISM {pa:.3f} vs Mamba {ma:.3f} → {w}")
    print(f"PRISM 승: {pw}/{len(results)} 난이도")


if __name__ == "__main__":
    main()
