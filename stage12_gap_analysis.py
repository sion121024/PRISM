"""
Stage 12: Mamba 격차 원인 분석 + 개선 실험

현재 결과 (Stage 10):
  PRISM-slim-K4 ≈ 13.0 ppl  vs  Mamba ≈ 6.6 ppl  (−6.4 ppl 격차)

가설 1: block_size=64는 너무 짧다
  Mamba: 입력 의존 선택적 SSM → 짧은 context에서 강함
  PRISM: Hebbian 연상기억 → 긴 context에서 패턴 축적 시 유리

가설 2: K=4는 충분히 깊지 않다
  더 많은 내부 반복 = 더 깊은 에너지 하강 → 더 나은 상태 추론

가설 3: 메모리 랭크가 너무 낮다 (rank=24)
  연상 기억 용량 부족 → 더 많은 슬롯 필요

실험:
  A) PRISM-slim-K4   block=64,  K=4, rank=24   (baseline, Stage 10 재현)
  B) PRISM-K8        block=64,  K=8, rank=24   (가설 2: 더 깊은 사고)
  C) PRISM-longctx   block=256, K=4, rank=24   (가설 1: 긴 context)
  D) PRISM-highrank  block=64,  K=4, rank=48   (가설 3: 더 큰 메모리)
  E) Mamba           block=64                   (참조)
  F) Mamba-longctx   block=256                  (긴 context Mamba 참조)
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


def make_prism(vocab_size, d, K, mem_rank, dec_hidden):
    return PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=64, K=K,
        decoder="mlp", dec_hidden=dec_hidden,
        simple_prior=True, use_urec=True,
        mem_rank=mem_rank, mem_scale=4.0,
    )


def fit_prism_params(vocab_size, target_params=55000, K=4, mem_rank=24):
    """target_params에 가장 가까운 d, dec_hidden 찾기 (mem_rank는 params에 무관)."""
    for d in range(60, 300, 4):
        dec_hidden = max(16, d // 4)
        m = make_prism(vocab_size, d, K, mem_rank, dec_hidden)
        n = count_params(m)
        if n >= target_params:
            return d, dec_hidden, n
    return 168, 42, -1


def train_one(model, train_ds, val_ds, args, name, block_size=None):
    bs = block_size or args.block_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = model.to(device)
    n_params = count_params(model)
    print(f"\n=== {name}: {n_params:,} params | block={bs} ===")

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
    p.add_argument("--skip_mamba", action="store_true")
    p.add_argument("--skip_longctx", action="store_true")
    args = p.parse_args()

    vocab_size = 65  # TinyShakespeare fixed
    results = {}

    # --- A) PRISM-slim baseline (block=64) ---
    train_ds = TinyShakespeare(block_size=64, split="train")
    val_ds   = TinyShakespeare(block_size=64, split="val")
    print(f"vocab={vocab_size}")

    slimA = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["PRISM-slim-K4"] = train_one(
        slimA, train_ds, val_ds, args, "PRISM-slim-K4", block_size=64)

    # --- B) PRISM-K8: 더 깊은 사고 (block=64) ---
    # 파라미터 중립: K=8 params 동일, 단순히 K 증가
    slimB = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=8,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    results["PRISM-slim-K8"] = train_one(
        slimB, train_ds, val_ds, args, "PRISM-slim-K8", block_size=64)

    # --- C) PRISM-longctx: 긴 context (block=256) ---
    if not args.skip_longctx:
        train_ds_long = TinyShakespeare(block_size=256, split="train")
        val_ds_long   = TinyShakespeare(block_size=256, split="val")
        slimC = PRISMLangModel(
            vocab_size=vocab_size, d=168, emb_dim=64, K=4,
            decoder="mlp", dec_hidden=42,
            simple_prior=True, use_urec=True,
            mem_rank=24, mem_scale=4.0,
        )
        results["PRISM-longctx-K4"] = train_one(
            slimC, train_ds_long, val_ds_long, args, "PRISM-longctx-K4", block_size=256)

    # --- D) PRISM-highrank: 더 큰 메모리 (rank=48) ---
    # mem_rank는 파라미터가 아닌 상태 크기 → params 동일, 기억 용량 2배
    slimD = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=48, mem_scale=4.0,  # 더 많은 메모리 슬롯
    )
    results["PRISM-highrank-K4"] = train_one(
        slimD, train_ds, val_ds, args, "PRISM-highrank-K4", block_size=64)

    # --- E) Mamba baseline ---
    if not args.skip_mamba:
        mamba = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
        results["Mamba"] = train_one(
            mamba, train_ds, val_ds, args, "Mamba", block_size=64)

        # --- F) Mamba-longctx ---
        if not args.skip_longctx:
            mamba_long = MambaLangModel(vocab_size=vocab_size, d_model=80, d_state=8)
            results["Mamba-longctx"] = train_one(
                mamba_long, train_ds_long, val_ds_long, args, "Mamba-longctx", block_size=256)

    print("\n" + "=" * 64)
    print("Stage 12: Mamba 격차 분석 (낮을수록 좋음)")
    print("=" * 64)
    for name, (ppl, n) in results.items():
        print(f"  {name:<22s}: val_ppl {ppl:.3f}  ({n:,} params)")

    base_prism = results.get("PRISM-slim-K4", (None,))[0]
    base_mamba = results.get("Mamba", (None,))[0]

    if base_prism and base_mamba:
        print(f"\n  초기 격차 (slim-K4 vs Mamba): {base_prism - base_mamba:+.3f} ppl")

    if base_prism:
        for name in ["PRISM-slim-K8", "PRISM-longctx-K4", "PRISM-highrank-K4"]:
            if name in results:
                diff = base_prism - results[name][0]
                print(f"  {name} 개선: {diff:+.3f} ppl vs baseline")

    if base_mamba and "Mamba-longctx" in results:
        diff_m = base_mamba - results["Mamba-longctx"][0]
        diff_p = (results.get("PRISM-longctx-K4", (base_prism,))[0] or base_prism) - (base_prism or 0)
        print(f"\n  긴 context Mamba 효과: {diff_m:+.3f} ppl")
        if "PRISM-longctx-K4" in results:
            diff_p = base_prism - results["PRISM-longctx-K4"][0]
            print(f"  긴 context PRISM 효과: {diff_p:+.3f} ppl")
            print(f"  (PRISM longctx 효과 - Mamba longctx 효과: {diff_p - diff_m:+.3f})")


if __name__ == "__main__":
    main()
