"""
Stage 10: PRISM 단순화 — 설계 단순화 + 속도 + 성능.

비교:
  PRISM-v1   : 기존 (use_prior, use_urec, dec_hidden=34) — 55K
  PRISM-slim : 단순화 (simple_prior, no urec, d↑, rank↑) — 55K
  PRISM-slim-K4: slim + K=4
  Mamba      : 참조 (54K)

단순화 원칙:
  - u_rec 제거 (16K params 절약): 원시 임베딩 직접 사용
  - prior_mu MLP 제거 (18K 절약): μ = x_prev (identity prior, 파라미터 없음)
  - 절약된 34K → d↑ + mem_rank↑ (더 큰 상태, 더 나은 기억)

실행:
  python stage10_slim.py --epochs 10 --block_size 64
"""

import argparse
import math
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare
from baselines import MambaLangModel


def train_one(model, train_loader, val_loader, args, name):
    device = torch.device(args.device)
    model = model.to(device)
    n_params = model.num_params()
    print(f"\n=== {name}: {n_params:,} params ===")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    best_ppl = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        t0 = time.time()
        for tokens in train_loader:
            tokens = tokens.to(device)
            out = model(tokens)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += out["loss"].item()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for tokens in val_loader:
                tokens = tokens.to(device)
                out = model(tokens)
                vloss += out["loss"].item()
        vppl = math.exp(vloss / len(val_loader))
        best_ppl = min(best_ppl, vppl)
        print(f"[{name}] epoch {epoch:2d} | "
              f"train_loss {total/len(train_loader):.4f} | "
              f"val_ppl {vppl:.3f} | {time.time()-t0:.0f}s")

    print(f"[{name}] BEST: {best_ppl:.3f}  ({n_params:,} params)")
    return best_ppl, n_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip_mamba", action="store_true")
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    print(f"vocab={vocab_size} | block_size={args.block_size}")

    results = {}

    # v1: 기존 (Stage 8/9 설정)
    v1 = PRISMLangModel(
        vocab_size=vocab_size, d=136, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=34,
        use_prior=True, use_urec=True,
        mem_rank=16, mem_scale=4.0,
    )
    results["PRISM-v1-K4"] = train_one(v1, train_loader, val_loader, args, "PRISM-v1-K4")

    # slim K=2: prior_mu MLP 제거만 (identity prior), u_rec 유지 → 절약 18K → d↑ rank↑
    # 교훈: u_rec 없이는 학습 실패 (u_rec이 시퀀스 컨텍스트 핵심)
    slim2 = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=2,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["PRISM-slim-K2"] = train_one(slim2, train_loader, val_loader, args, "PRISM-slim-K2")

    # slim K=4: 단순화 + K 증가
    slim4 = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["PRISM-slim-K4"] = train_one(slim4, train_loader, val_loader, args, "PRISM-slim-K4")

    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    print("\n" + "=" * 64)
    print("Stage 10: 단순화 효과 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<18s}: val_ppl {ppl:.3f}  ({n:,} params)")

    if "PRISM-v1-K4" in results and "PRISM-slim-K4" in results:
        diff = results["PRISM-v1-K4"][0] - results["PRISM-slim-K4"][0]
        print(f"\n  단순화 효과 (v1→slim): {diff:+.3f} ppl")
    if "PRISM-slim-K2" in results and "PRISM-slim-K4" in results:
        keff = results["PRISM-slim-K2"][0] - results["PRISM-slim-K4"][0]
        print(f"  K-effect (slim K2→K4): +{keff:.3f} ppl")
    if "PRISM-slim-K4" in results and "Mamba" in results:
        gap = results["PRISM-slim-K4"][0] - results["Mamba"][0]
        print(f"  PRISM-slim vs Mamba: {gap:+.3f} ppl")


if __name__ == "__main__":
    main()
