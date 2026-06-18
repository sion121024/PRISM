"""
Stage 14b: Remaining models from Stage 14 (slim+all-K4, slim+conv-K4, slim+conv+all-K4, Mamba)
Already have: slim-K4 (28.156), slim+sel-K4 (26.070), slim+mom-K4 (28.103)
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
    print("(Already have: slim-K4=28.156, slim+sel-K4=26.070, slim+mom-K4=28.103)")

    results = {}

    # D) sel + mom + prior_bias (no conv)
    results["slim+all-K4"] = train_one(
        make_prism(vocab_size, input_dep_pi=True, prior_bias=True, momentum=0.9),
        train_loader, val_loader, args, "slim+all-K4")

    # E) conv only (THE KEY TEST)
    results["slim+conv-K4"] = train_one(
        make_prism(vocab_size, use_conv=True),
        train_loader, val_loader, args, "slim+conv-K4")

    # F) conv + all
    results["slim+conv+all-K4"] = train_one(
        make_prism(vocab_size, use_conv=True, input_dep_pi=True,
                   prior_bias=True, momentum=0.9),
        train_loader, val_loader, args, "slim+conv+all-K4")

    # G) Mamba reference
    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    print("\n" + "=" * 64)
    print("Stage 14b: Remaining results (낮을수록 좋음)")
    print("=" * 64)
    print("  slim-K4                   : val_ppl 28.156  (55,594 params)  [previous]")
    print("  slim+sel-K4               : val_ppl 26.070  (70,674 params)  [previous]")
    print("  slim+mom-K4               : val_ppl 28.103  (55,594 params)  [previous]")
    for name, (ppl, n) in results.items():
        print(f"  {name:<26s}: val_ppl {ppl:.3f}  ({n:,} params)")

    base = 28.156
    for name, (ppl, _) in results.items():
        diff = base - ppl
        print(f"  {name} vs slim기준: {diff:+.3f} ppl")
    if "Mamba" in results:
        mamba_ppl = results["Mamba"][0]
        print(f"\n  slim기준 vs Mamba: {base - mamba_ppl:+.3f} ppl")
        if "slim+conv+all-K4" in results:
            all_ppl = results["slim+conv+all-K4"][0]
            total_gap = base - mamba_ppl
            closed = base - all_ppl
            if total_gap != 0:
                print(f"  slim+conv+all 격차 감소: {100*closed/abs(total_gap):.1f}% of gap")


if __name__ == "__main__":
    main()
