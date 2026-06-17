"""
Stage 9: PRISM vs Mamba vs LSTM — 파라미터 매칭 비교.

동일 파라미터 예산 (~54-56K)에서 공정 비교:
  PRISM-K2  : prior + u_rec, K=2
  PRISM-K4  : prior + u_rec, K=4
  Mamba     : S6 selective SSM (순수 PyTorch)
  LSTM      : 1-layer LSTM

핵심 질문:
  - PRISM K-effect가 Mamba 대비 어떤 수준인가?
  - 에너지 기반 아키텍처의 경쟁력

실행:
  python stage9_mamba_compare.py --epochs 10 --block_size 64
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
from baselines import LSTMLangModel, MambaLangModel


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
    p.add_argument("--skip_lstm", action="store_true")
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    print(f"vocab={vocab_size} | train={len(train_ds)} val={len(val_ds)}"
          f" | block_size={args.block_size}")

    results = {}

    def make_prism(K):
        # d=136, emb_dim=64, dec_hidden=34 → ~55,452 params (stage8 동일 설정)
        return PRISMLangModel(
            vocab_size=vocab_size, d=136, emb_dim=64, K=K,
            decoder="mlp", dec_hidden=34,
            use_prior=True, use_urec=True,
            mem_rank=16, mem_scale=4.0,
            memory_mode="sliding",
        )

    # PRISM K=2
    results["PRISM-K2"] = train_one(make_prism(2), train_loader, val_loader, args, "PRISM-K2")

    # PRISM K=4
    results["PRISM-K4"] = train_one(make_prism(4), train_loader, val_loader, args, "PRISM-K4")

    # Mamba (d_model=80, d_state=8 → 54,400 params)
    mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
    results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    if not args.skip_lstm:
        # LSTM: emb=64, hidden=80 → ~56,080 params
        lstm = LSTMLangModel(vocab_size=vocab_size, emb_dim=64, hidden_dim=80, n_layers=1)
        results["LSTM"] = train_one(lstm, train_loader, val_loader, args, "LSTM")

    print("\n" + "=" * 62)
    print("Stage 9: PRISM vs Mamba vs LSTM (낮을수록 좋음)")
    print("=" * 62)
    for name, (ppl, n) in results.items():
        print(f"  {name:<12s}: val_ppl {ppl:.3f}  ({n:,} params)")

    if "PRISM-K4" in results and "Mamba" in results:
        diff = results["PRISM-K4"][0] - results["Mamba"][0]
        print(f"\n  PRISM-K4 vs Mamba: {diff:+.3f} ppl "
              f"({'PRISM 승' if diff < 0 else 'Mamba 승'})")
    if "PRISM-K2" in results and "PRISM-K4" in results:
        k_eff = results["PRISM-K2"][0] - results["PRISM-K4"][0]
        print(f"  K-effect (K2→K4): +{k_eff:.3f} ppl")


if __name__ == "__main__":
    main()
