"""
Stage 13: 입력 의존 precision (Selective PRISM) 효과 검증

가설: Π1(u), Π2(u) — 각 토큰에 맞게 지각/기억 precision을 조절하면
      Mamba 선택적 메커니즘과 유사한 효과로 성능이 향상된다.

이론적 근거:
  Mamba: B(x_t), C(x_t) — 입력 의존 상태 행렬
  PRISM: Π1 = const, Π2 = const — 모든 토큰에 동일한 precision
  Selective PRISM: Π1(u), Π2(u) — 토큰별 precision (Mamba의 에너지 유사체)

비교:
  A) PRISM-slim-K4        : d=168, identity prior, Π=const   (기준)
  B) PRISM-selective-K4   : d=168, identity prior, Π=Π(u)    (파라미터 많음)
  C) PRISM-sel-matched-K4 : d≈130, identity prior, Π=Π(u)    (파라미터 맞춤)
  D) Mamba                : d_model=80                        (참조)

10 epoch 비교.
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


def find_d_for_params(vocab_size, target, emb_dim=64, input_dep_pi=False, simple_prior=True):
    """target params에 가장 가까운 d 찾기."""
    for d in range(60, 300, 2):
        dec_hidden = max(16, d // 4)
        m = PRISMLangModel(
            vocab_size=vocab_size, d=d, emb_dim=emb_dim, K=4,
            decoder="mlp", dec_hidden=dec_hidden,
            simple_prior=simple_prior, use_urec=True,
            mem_rank=24, mem_scale=4.0,
            input_dep_pi=input_dep_pi,
        )
        n = count_params(m)
        if n >= target:
            return d, dec_hidden, n
    return 168, 42, -1


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

    # A) PRISM-slim-K4: 기준 (Π=const)
    baseline = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
        input_dep_pi=False,
    )
    results["slim-K4"] = train_one(baseline, train_loader, val_loader, args, "slim-K4")
    target_params = results["slim-K4"][1]

    # B) PRISM-selective-K4: Π(u) — 더 많은 파라미터 (d=168 유지)
    selective = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
        input_dep_pi=True,
    )
    results["selective-K4"] = train_one(selective, train_loader, val_loader, args, "selective-K4")

    # C) PRISM-sel-matched-K4: Π(u), 파라미터 기준선과 맞춤
    d_m, dec_h_m, n_m = find_d_for_params(vocab_size, target_params, input_dep_pi=True)
    print(f"\n파라미터 맞춤: d={d_m}, dec_hidden={dec_h_m}, params={n_m:,}")
    sel_matched = PRISMLangModel(
        vocab_size=vocab_size, d=d_m, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=dec_h_m,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
        input_dep_pi=True,
    )
    results["sel-matched-K4"] = train_one(sel_matched, train_loader, val_loader, args, "sel-matched-K4")

    # D) Mamba: 참조
    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(mamba, train_loader, val_loader, args, "Mamba")

    print("\n" + "=" * 64)
    print("Stage 13: Selective PRISM (Π(u)) 효과 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<22s}: val_ppl {ppl:.3f}  ({n:,} params)")

    base_ppl = results.get("slim-K4", (None,))[0]
    if base_ppl:
        for name in ["selective-K4", "sel-matched-K4"]:
            if name in results:
                diff = base_ppl - results[name][0]
                print(f"  {name} 개선: {diff:+.3f} ppl vs 기준")
        if "Mamba" in results:
            gap = base_ppl - results["Mamba"][0]
            print(f"\n  slim vs Mamba 격차: {gap:+.3f} ppl")
            if "sel-matched-K4" in results:
                sel_gap = results["sel-matched-K4"][0] - results["Mamba"][0]
                print(f"  sel-matched vs Mamba 격차: {sel_gap:+.3f} ppl")
                closed = gap - sel_gap
                print(f"  격차 감소: {closed:+.3f} ppl ({100*closed/abs(gap):.1f}%)")


if __name__ == "__main__":
    main()
