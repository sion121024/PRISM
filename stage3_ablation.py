"""
Stage 3 Ablation: 방향 1+3 검증.

  방향 1: 메모리 강화 (scale 1/√d → 1.0, rank 16 → 32)
  방향 3: carry gate — 토큰 사이 비선형 상태 전이

비교 구성:
  A. baseline    — linear, mem_scale=1/√d=0.0625, rank=16, no carry  (=Stage2 원본)
  B. mem_boost   — linear, mem_scale=1.0, rank=32, no carry          (방향 1)
  C. mem+carry   — linear, mem_scale=1.0, rank=32, carry_nonlin      (방향 1+3)
  LSTM           — param-matched baseline

실행:
  python stage3_ablation.py --epochs 5 --block_size 64
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


def make_prism(vocab_size, d, emb_dim, K, alpha, mem_rank, mem_scale, carry_nonlin, decoder="linear", state_norm=None):
    return PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=emb_dim,
        K=K, alpha=alpha, memory_mode="sliding",
        mem_rank=mem_rank, mem_scale=mem_scale,
        decoder=decoder, dec_hidden=64,
        carry_nonlin=carry_nonlin,
        state_norm=state_norm,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--K", type=int, default=2)
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

    # mem_scale = spectral_norm(M) 근사 (SlidingMemory.scale = mem_scale/rank)
    # 안정성 조건: alpha*(1+mem_scale)²*pi2_max < 2  →  mem_scale < 6.6 (alpha=0.05)
    configs = [
        # name,              ms,   rank, carry,  dec,      sn,    K
        ("A-baseline",       1.0,  16,   False,  "linear", None,  2),
        ("B-mem_boost",      4.0,  32,   False,  "linear", None,  2),
        ("C-mem+carry",      4.0,  32,   True,   "linear", None,  2),
        ("D-mlp+carry",      4.0,  32,   True,   "mlp",    False, 2),  # carry+mlp decoder, K=2
        ("E-mlp+carry-K4",   4.0,  32,   True,   "mlp",    False, 4),  # carry+mlp decoder, K=4
    ]

    results = {}
    for name, mem_scale, mem_rank, carry_nonlin, decoder, state_norm, K in configs:
        model = make_prism(
            vocab_size, args.d, args.emb_dim, K, args.alpha,
            mem_rank, mem_scale, carry_nonlin, decoder, state_norm,
        )
        results[name] = train_one(model, train_loader, val_loader, args, name)

    if not args.skip_lstm:
        lstm = LSTMLangModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim,
            hidden_dim=args.lstm_hidden, n_layers=1,
        )
        results["LSTM"] = train_one(lstm, train_loader, val_loader, args, "LSTM")

    print("\n" + "=" * 56)
    print("Stage 3 Ablation 결과 (낮을수록 좋음)")
    print("=" * 56)
    for name, (ppl, n) in results.items():
        print(f"  {name:<16s}: val_ppl {ppl:.3f}  ({n:,} params)")
    print()
    print("방향 1 효과: A vs B (메모리 강화)")
    print("방향 3 효과: B vs C (carry gate 추가)")
    print("방향 4 효과: C vs D (mlp decoder 추가)")
    print("방향 5 효과: D vs E (K=4 증가)")


if __name__ == "__main__":
    main()
