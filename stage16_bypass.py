"""
Stage 16: Bypass — n-gram shortcut 효과 검증

use_bypass=True: logits += W_bypass · u_conv
  - 에너지 상태와 독립적으로 로컬 n-gram 패턴을 직접 예측.
  - u_conv = SiLU(conv1d(embed)) 이므로 use_conv=True 와 함께 사용 시
    4-gram 컨텍스트를 바로 어휘 분포에 매핑.
  - W_bypass: Linear(emb_dim, vocab_size), zeros-init → 학습 전 효과 없음.

Stage 14 best (slim+conv-K4) 기준으로:
  A) slim+conv-K4           : Stage 14 참조 기준
  B) slim+conv+bypass-K4    : bypass only (gate 없음)
  C) slim+conv+gate+bypass-K4        : bypass + gate
  D) slim+conv+gate+bypass+all-K4    : 전체 (bypass+gate+sel+mom+prior_bias)
  E) Mamba                  : 참조

5 epoch, TinyShakespeare char-LM.
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

    # A) conv 기준 (Stage 14 참조)
    results["slim+conv-K4"] = train_one(
        make_prism(vocab_size, use_conv=True),
        train_loader, val_loader, args, "slim+conv-K4")

    # B) conv + bypass
    results["slim+conv+bypass-K4"] = train_one(
        make_prism(vocab_size, use_conv=True, use_bypass=True),
        train_loader, val_loader, args, "slim+conv+bypass-K4")

    # C) conv + gate + bypass
    results["slim+conv+gate+bypass-K4"] = train_one(
        make_prism(vocab_size, use_conv=True, use_gate=True, use_bypass=True),
        train_loader, val_loader, args, "slim+conv+gate+bypass-K4")

    # D) 전체: conv + gate + bypass + sel + mom + prior_bias
    results["slim+conv+gate+bypass+all-K4"] = train_one(
        make_prism(vocab_size, use_conv=True, use_gate=True, use_bypass=True,
                   input_dep_pi=True, prior_bias=True, momentum=0.9),
        train_loader, val_loader, args, "slim+conv+gate+bypass+all-K4")

    # E) Mamba
    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    print("\n" + "=" * 64)
    print("Stage 16: Bypass 효과 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<36s}: val_ppl {ppl:.3f}  ({n:,} params)")

    base = results.get("slim+conv-K4", (None,))[0]
    if base:
        for name in ["slim+conv+bypass-K4", "slim+conv+gate+bypass-K4",
                     "slim+conv+gate+bypass+all-K4"]:
            if name in results:
                diff = base - results[name][0]
                print(f"  {name} vs conv기준: {diff:+.3f} ppl")
        if "Mamba" in results:
            mamba_ppl = results["Mamba"][0]
            print(f"\n  conv기준 vs Mamba: {base - mamba_ppl:+.3f} ppl")
            if "slim+conv+gate+bypass+all-K4" in results:
                all_ppl = results["slim+conv+gate+bypass+all-K4"][0]
                print(f"  최선 PRISM vs Mamba: {all_ppl:.3f} vs {mamba_ppl:.3f} "
                      f"(gap {all_ppl - mamba_ppl:+.3f})")


if __name__ == "__main__":
    main()
