"""
Stage 8: 파라미터 매칭 비교.

PRISM(d=176, ~54K) vs LSTM(~56K) — 동일 파라미터 예산에서 공정 비교.
설계 정합 구성: carry gate 없음 | prior 항 | 비대칭 Hebbian | MLP decoder

핵심 질문:
  - 같은 파라미터에서 PRISM이 LSTM을 이길 수 있는가?
  - K-effect (K2 vs K4)는 파라미터 효율에 기여하는가?

실행:
  python stage8_param_match.py --epochs 10 --block_size 64
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
from baselines import LSTMLangModel


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

    print(f"[{name}] BEST val_ppl: {best_ppl:.3f}  ({n_params:,} params)")
    return best_ppl, n_params


def make_prism_matched(vocab_size, d, emb_dim, K, alpha, dec_hidden, mem_rank=16, mem_scale=4.0):
    """파라미터 매칭 PRISM: carry gate 없음, prior 항, u_rec 없음 (~54K)."""
    return PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=emb_dim,
        K=K, alpha=alpha, memory_mode="sliding",
        mem_rank=mem_rank, mem_scale=mem_scale,
        decoder="mlp", dec_hidden=dec_hidden,
        carry_nonlin=False,
        state_norm=False,
        use_prior=True,
        use_urec=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d", type=int, default=176)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--dec_hidden", type=int, default=44)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lstm_hidden", type=int, default=80)
    p.add_argument("--skip_lstm", action="store_true")
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    print(f"vocab={vocab_size} | train={len(train_ds)} val={len(val_ds)} chunks"
          f" | block_size={args.block_size}")
    print("파라미터 매칭: PRISM ~54K vs LSTM ~56K")

    results = {}

    # K=2 vs K=4 비교
    for K in [2, 4]:
        name = f"PRISM-prior-K{K}"
        model = make_prism_matched(
            vocab_size, args.d, args.emb_dim, K, args.alpha, args.dec_hidden,
        )
        results[name] = train_one(model, train_loader, val_loader, args, name)

    if not args.skip_lstm:
        lstm = LSTMLangModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim,
            hidden_dim=args.lstm_hidden, n_layers=1,
        )
        results["LSTM"] = train_one(lstm, train_loader, val_loader, args, "LSTM")

    print("\n" + "=" * 62)
    print("Stage 8: 파라미터 매칭 결과 (낮을수록 좋음)")
    print("=" * 62)
    for name, (ppl, n) in results.items():
        print(f"  {name:<22s}: val_ppl {ppl:.3f}  ({n:,} params)")

    ppls = {n: p for n, (p, _) in results.items()}
    k2, k4 = ppls.get("PRISM-prior-K2"), ppls.get("PRISM-prior-K4")
    lstm = ppls.get("LSTM")
    if k2 and k4:
        print(f"\n  K-effect (K2→K4): {k2-k4:+.3f} ppl")
    if k4 and lstm:
        winner = "PRISM" if k4 < lstm else "LSTM"
        print(f"  PRISM-K4 vs LSTM: {k4-lstm:+.3f} ppl  ({winner} 승)")


if __name__ == "__main__":
    main()
