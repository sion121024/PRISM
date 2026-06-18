"""
Stage 10c: prior_bias 효과 검증

가설: prior_bias=True 는 identity prior의 느린 초기 학습을 해결하면서
      K-effect는 유지 (심지어 MLP prior보다 클 수 있음)

비교:
  A) v1-K4         : learned prior_mu MLP (18K params)  → 기준
  B) slim-K4       : identity prior (0 params)          → 느린 수렴
  C) slim+bias-K4  : biased prior (d params)            → 빠른 수렴?
  D) slim+bias-K2  : biased prior K=2                   → K-effect 측정용

10 epoch 로 수렴 비교.
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


def train_one(model, train_loader, val_loader, args, name):
    device = torch.device(args.device)
    model = model.to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name}: {n:,} params ===")

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

    print(f"[{name}] BEST: {best_ppl:.3f}  ({n:,} params)")
    return best_ppl, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    print(f"vocab={vocab_size} | block_size={args.block_size}")

    results = {}

    # A) v1-K4: 기준 (learned prior_mu MLP)
    v1 = PRISMLangModel(
        vocab_size=vocab_size, d=136, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=34,
        use_prior=True, use_urec=True,
        mem_rank=16, mem_scale=4.0,
    )
    results["v1-K4"] = train_one(v1, train_loader, val_loader, args, "v1-K4")

    # B) slim-K4: identity prior (d=168, 0 prior params)
    slim = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, prior_bias=False, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["slim-K4"] = train_one(slim, train_loader, val_loader, args, "slim-K4")

    # C) slim+bias-K4: biased prior (d=168, d prior params = 168 추가)
    slim_b4 = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, prior_bias=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["slim+bias-K4"] = train_one(slim_b4, train_loader, val_loader, args, "slim+bias-K4")

    # D) slim+bias-K2: K-effect 측정
    slim_b2 = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=2,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, prior_bias=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["slim+bias-K2"] = train_one(slim_b2, train_loader, val_loader, args, "slim+bias-K2")

    print("\n" + "=" * 64)
    print("Stage 10c: prior_bias 효과 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<18s}: val_ppl {ppl:.3f}  ({n:,} params)")

    if "v1-K4" in results and "slim+bias-K4" in results:
        diff = results["v1-K4"][0] - results["slim+bias-K4"][0]
        print(f"\n  biased prior vs v1: {diff:+.3f} ppl (양수=biased가 나쁨)")
    if "slim-K4" in results and "slim+bias-K4" in results:
        diff = results["slim-K4"][0] - results["slim+bias-K4"][0]
        print(f"  bias 효과 (slim→slim+bias): {diff:+.3f} ppl")
    if "slim+bias-K2" in results and "slim+bias-K4" in results:
        diff = results["slim+bias-K2"][0] - results["slim+bias-K4"][0]
        print(f"  K-effect (biased K2→K4): {diff:+.3f} ppl")


if __name__ == "__main__":
    main()
