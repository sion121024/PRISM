"""
Stage 2: PRISM vs LSTM 문자 LM perplexity 비교.

공정한 비교 원칙:
  - 동일 데이터 (shared vocab train/val split)
  - 파라미터 예산 매칭 (~55K)
  - 동일 학습 설정 (lr, epochs, batch)

PRISM의 논지는 "적은 파라미터로 경쟁"이므로 파라미터를 맞추고 ppl을 본다.

실행:
  python stage2_compare.py --epochs 15 --block_size 64
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


def train_one(model, train_loader, val_loader, vocab_size, args, name, tbptt):
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
            out = model(tokens, tbptt_window=tbptt) if tbptt is not None \
                else model(tokens)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    # PRISM
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mem_rank", type=int, default=16)
    p.add_argument("--tbptt", type=int, default=0)
    p.add_argument("--decoder", choices=["linear", "mlp"], default="linear")
    p.add_argument("--dec_hidden", type=int, default=64)
    p.add_argument("--skip_lstm", action="store_true", default=False)
    # LSTM (param-matched)
    p.add_argument("--lstm_hidden", type=int, default=80)
    p.add_argument("--lstm_layers", type=int, default=1)
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    print(f"vocab={vocab_size} | train={len(train_ds)} val={len(val_ds)} chunks"
          f" | block_size={args.block_size}")

    results = {}

    pname = f"PRISM-{args.decoder}"
    prism = PRISMLangModel(
        vocab_size=vocab_size, d=args.d, emb_dim=args.emb_dim,
        K=args.K, alpha=args.alpha, memory_mode="sliding", mem_rank=args.mem_rank,
        decoder=args.decoder, dec_hidden=args.dec_hidden,
    )
    results[pname] = train_one(
        prism, train_loader, val_loader, vocab_size, args, pname, args.tbptt)

    if not args.skip_lstm:
        lstm = LSTMLangModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim,
            hidden_dim=args.lstm_hidden, n_layers=args.lstm_layers,
        )
        results["LSTM"] = train_one(
            lstm, train_loader, val_loader, vocab_size, args, "LSTM", None)

    print("\n" + "=" * 50)
    print("Stage 2 결과 (낮을수록 좋음)")
    print("=" * 50)
    for name, (ppl, n) in results.items():
        print(f"  {name:6s}: val_ppl {ppl:.3f}  ({n:,} params)")


if __name__ == "__main__":
    main()
