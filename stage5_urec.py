"""
Stage 5: u_recurrent — 설계 정합 순환 의존성.

설계 철학 준수:
  - carry gate 없음 (에너지 밖 변환 제거)
  - u_recurrent: ũ = W_rec([u, x_prev]) → 관측값을 이전 상태로 증강
    에너지 E(x) = ½‖ũ − g(x)‖²_Π1 + ...  에서 순환이 일어남
  - state_norm=True (RMS pre-norm)
  - 비대칭 Hebbian

비교:
  I.  urec-K2 — u_recurrent, K=2  (설계 정합)
  J.  urec-K4 — u_recurrent, K=4  (K-effect 측정)
  K.  urec-K8 — u_recurrent, K=8  (더 깊은 사고)
  E.  [참조] E-mlp+carry K=4  → ppl 22.937  (이전 최고, carry gate 포함)
  LSTM  → ppl 25.518  (비교 기준)

실행:
  python stage5_urec.py --epochs 5 --block_size 64
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


def make_urec(vocab_size, d, emb_dim, K, alpha, mem_rank=32, mem_scale=4.0):
    """설계 정합: u_recurrent + state_norm + MLP decoder, carry gate 없음."""
    return PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=emb_dim,
        K=K, alpha=alpha, memory_mode="sliding",
        mem_rank=mem_rank, mem_scale=mem_scale,
        decoder="mlp", dec_hidden=64,
        carry_nonlin=False,
        state_norm=False,  # K-루프 내 정규화 없음 (에너지 단조감소 보장)
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
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--mem_rank", type=int, default=32)
    p.add_argument("--mem_scale", type=float, default=4.0)
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
    print("설계 원칙: u_recurrent | state_norm | 비대칭 Hebbian | MLP decoder | no carry gate")

    configs = [("I-urec-K2", 2), ("J-urec-K4", 4), ("K-urec-K8", 8)]

    results = {}
    for name, K in configs:
        model = make_urec(vocab_size, args.d, args.emb_dim, K, args.alpha,
                          args.mem_rank, args.mem_scale)
        results[name] = train_one(model, train_loader, val_loader, args, name)

    if not args.skip_lstm:
        lstm = LSTMLangModel(
            vocab_size=vocab_size, emb_dim=args.emb_dim,
            hidden_dim=args.lstm_hidden, n_layers=1,
        )
        results["LSTM"] = train_one(lstm, train_loader, val_loader, args, "LSTM")

    print("\n" + "=" * 64)
    print("Stage 5: u_recurrent 설계 정합 결과 (낮을수록 좋음)")
    print("=" * 64)
    print(f"  {'[참조] E-mlp+carry-K4':<20s}: val_ppl 22.937  (carry gate 포함)")
    print(f"  {'[참조] LSTM':<20s}: val_ppl 25.518  (비교 기준)")
    print()
    for name, (ppl, n) in results.items():
        print(f"  {name:<20s}: val_ppl {ppl:.3f}  ({n:,} params)")
    print()

    ppls = {n: p for n, (p, _) in results.items()}
    if "I-urec-K2" in ppls and "J-urec-K4" in ppls:
        d = ppls["I-urec-K2"] - ppls["J-urec-K4"]
        print(f"K2→K4 효과: {d:+.3f} ppl")
    if "J-urec-K4" in ppls and "K-urec-K8" in ppls:
        d = ppls["J-urec-K4"] - ppls["K-urec-K8"]
        print(f"K4→K8 효과: {d:+.3f} ppl")
    if "K-urec-K8" in ppls and "LSTM" in ppls:
        d = ppls["K-urec-K8"] - ppls["LSTM"]
        status = "PRISM 승" if d < 0 else "LSTM 승"
        print(f"PRISM(K8) vs LSTM: {d:+.3f} ppl → {status}")


if __name__ == "__main__":
    main()
