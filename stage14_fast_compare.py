"""
Stage 14: 빠른 비교 — 누적된 모든 개선사항 통합 효과 검증

이전 단계에서 구현된 개선사항:
  1. input_dep_pi=True  — Π1(u), Π2(u) 선택적 precision (Mamba 유사체)
  2. prior_bias=True    — μ = x_prev + b (빠른 수렴)
  3. momentum=0.9      — Heavy-ball K-step (빠른 수렴)
  4. use_conv=True      — Depthwise conv1d: 로컬 n-gram 패턴 (+320 params, Mamba 유사체)

비교:
  A) slim-K4             : 기준 (모든 기능 off)
  B) slim+sel-K4         : input_dep_pi only
  C) slim+mom-K4         : momentum=0.9 only
  D) slim+all-K4         : input_dep_pi + prior_bias + momentum
  E) slim+conv-K4        : use_conv only
  F) slim+conv+all-K4    : 전체 (conv + sel + mom + prior_bias)
  G) Mamba               : 참조

5 epoch (신호 충분, 런타임 합리적).
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one(model, train_loader, val_loader, args, name):
    device = torch.device(args.device)
    model = model.to(device)
    n = count_params(model)
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


def make_prism(vocab_size, **kwargs):
    defaults = dict(d=168, emb_dim=64, K=4, decoder="mlp", dec_hidden=42,
                    simple_prior=True, use_urec=True, mem_rank=24, mem_scale=4.0)
    defaults.update(kwargs)
    return PRISMLangModel(vocab_size=vocab_size, **defaults)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
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
    print(f"vocab={vocab_size} | block_size={args.block_size} | epochs={args.epochs}")

    results = {}

    # A) 기준: slim-K4 (모든 기능 off)
    results["slim-K4"] = train_one(
        make_prism(vocab_size), train_loader, val_loader, args, "slim-K4")

    # B) input_dep_pi only
    results["slim+sel-K4"] = train_one(
        make_prism(vocab_size, input_dep_pi=True),
        train_loader, val_loader, args, "slim+sel-K4")

    # C) momentum only (β=0.9)
    results["slim+mom-K4"] = train_one(
        make_prism(vocab_size, momentum=0.9),
        train_loader, val_loader, args, "slim+mom-K4")

    # D) 모든 기능 활성화 (conv 제외)
    results["slim+all-K4"] = train_one(
        make_prism(vocab_size, input_dep_pi=True, prior_bias=True, momentum=0.9),
        train_loader, val_loader, args, "slim+all-K4")

    # E) conv only: 로컬 n-gram 패턴 캡처 (Mamba conv1d 유사체, +320 params)
    results["slim+conv-K4"] = train_one(
        make_prism(vocab_size, use_conv=True),
        train_loader, val_loader, args, "slim+conv-K4")

    # F) 전체 누적: conv + sel + mom + prior_bias
    results["slim+conv+all-K4"] = train_one(
        make_prism(vocab_size, use_conv=True, input_dep_pi=True,
                   prior_bias=True, momentum=0.9),
        train_loader, val_loader, args, "slim+conv+all-K4")

    # G) Mamba: 참조
    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    print("\n" + "=" * 64)
    print("Stage 14: 누적 개선 효과 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<26s}: val_ppl {ppl:.3f}  ({n:,} params)")

    base = results.get("slim-K4", (None,))[0]
    if base:
        for name in ["slim+sel-K4", "slim+mom-K4", "slim+all-K4",
                     "slim+conv-K4", "slim+conv+all-K4"]:
            if name in results:
                diff = base - results[name][0]
                print(f"  {name} vs 기준: {diff:+.3f} ppl")
        if "Mamba" in results:
            print(f"\n  기준 vs Mamba: {base - results['Mamba'][0]:+.3f} ppl")
            if "slim+conv+all-K4" in results:
                all_ppl = results["slim+conv+all-K4"][0]
                closed = base - all_ppl
                total_gap = base - results["Mamba"][0]
                print(f"  slim+conv+all 격차 감소: {closed:+.3f} ppl ({100*closed/abs(total_gap):.1f}% of gap)")


if __name__ == "__main__":
    main()
