"""
Stage 11: 적응형 K(t) 실측 검증

고정 K vs 적응형 K(t):
  - 어려운 토큰(엔트로피 높음) → K 더 많이
  - 쉬운 토큰(엔트로피 낮음) → K 적게

검증 방법:
  1. PRISM-slim 학습 (고정 K=4, 10 epoch)
  2. 동일 모델로 eval:
     a) 고정 K=2, K=4, K=8
     b) 적응형 K (K_min=1, K_max=8)
  3. K 분포 통계: 평균 K, 엔트로피-K 상관
  4. ppl 비교 (같은 계산 예산 대비)

결론 목표:
  "어려운 토큰에 K 더 줘도 쉬운 토큰 K 줄여서 평균 유지 → ppl ↑ or same"
  → 이중시계의 핵심 가치 (K(t) 적응성) 실증
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare


def train_model(model, train_loader, val_loader, args, name):
    device = torch.device(args.device)
    model = model.to(device)
    n_params = model.num_params()
    print(f"\n=== 학습: {name} ({n_params:,} params) ===")

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
        print(f"  epoch {epoch:2d} | train_loss {total/len(train_loader):.4f} | "
              f"val_ppl {vppl:.3f} | {time.time()-t0:.0f}s")

    print(f"  BEST: {best_ppl:.3f}")
    return model, best_ppl


def eval_fixed_k(model, val_loader, device, K):
    """고정 K로 eval."""
    model.eval()
    original_K = model.cell.K
    model.cell.K = K
    vloss = 0.0
    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)
            out = model(tokens)
            vloss += out["loss"].item()
    ppl = math.exp(vloss / len(val_loader))
    model.cell.K = original_K
    return ppl


def eval_adaptive_k(model, val_loader, device, K_min, K_max):
    """
    적응형 K(t)로 eval.
    엔트로피 기반으로 토큰마다 K 결정.
    """
    model.eval()
    vloss = 0.0
    total_k = 0
    n_tokens = 0

    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)
            out = model(tokens, adaptive_K=True, K_min=K_min, K_max=K_max)
            vloss += out["loss"].item()
            if "k_used" in out:
                total_k += sum(out["k_used"])
                n_tokens += len(out["k_used"])

    ppl = math.exp(vloss / len(val_loader))
    avg_k = total_k / n_tokens if n_tokens > 0 else 0
    return ppl, avg_k


def compute_entropy_k_correlation(model, val_loader, device, K_min=1, K_max=8):
    """
    각 토큰의 엔트로피와 할당된 K의 상관관계 측정.
    """
    model.eval()
    entropies = []
    k_values = []

    B_sample = 4  # 빠른 측정용 샘플
    count = 0

    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens[:B_sample].to(device)
            T = tokens.shape[1]

            x, mem = model.init_state(B_sample, device)
            u_all = model.embed(tokens[:, :-1])
            prev_logits = None

            for t in range(T - 1):
                u_raw = u_all[:, t]
                if model.use_urec:
                    u = model.u_rec2(F.gelu(model.u_rec1(
                        torch.cat([u_raw, x], dim=-1))))
                else:
                    u = u_raw

                if prev_logits is not None:
                    probs = torch.softmax(prev_logits, dim=-1)
                    ent = -(probs * (probs + 1e-9).log()).sum(-1).max().item()
                    k_t = model._token_K(prev_logits, K_min, K_max)
                    entropies.append(ent)
                    k_values.append(k_t)

                x_prior = x if model.simple_prior else None
                x, mem = model.cell(u, mem, x, training=False,
                                    x_prior=x_prior, K=None)
                rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
                x = x * rms
                prev_logits = model.output_proj(x.detach())

            count += 1
            if count >= 10:
                break

    return entropies, k_values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    print(f"vocab={vocab_size} | block_size={args.block_size}")

    # Stage 10 slim-K4 설정 사용
    model = PRISMLangModel(
        vocab_size=vocab_size, d=168, emb_dim=64, K=4,
        decoder="mlp", dec_hidden=42,
        simple_prior=True, use_urec=True,
        mem_rank=24, mem_scale=4.0,
    )
    model, train_ppl = train_model(model, train_loader, val_loader, args, "PRISM-slim-K4")
    model = model.to(device)

    print("\n" + "=" * 64)
    print("Stage 11: 고정 K vs 적응형 K(t) 비교")
    print("=" * 64)

    # 고정 K 비교
    for K_fixed in [1, 2, 4, 8]:
        t0 = time.time()
        ppl = eval_fixed_k(model, val_loader, device, K=K_fixed)
        elapsed = time.time() - t0
        print(f"  고정  K={K_fixed}: val_ppl {ppl:.3f}  ({elapsed:.0f}s)")

    # 적응형 K 비교 (다양한 K_max)
    for k_min, k_max in [(1, 8), (2, 8), (1, 4)]:
        t0 = time.time()
        ppl, avg_k = eval_adaptive_k(model, val_loader, device, K_min=k_min, K_max=k_max)
        elapsed = time.time() - t0
        print(f"  적응 K=[{k_min},{k_max}]: val_ppl {ppl:.3f}  avg_K={avg_k:.2f}  ({elapsed:.0f}s)")

    # 엔트로피-K 상관 분석
    print("\n엔트로피-K 상관:")
    entropies, k_vals = compute_entropy_k_correlation(
        model, val_loader, device, K_min=1, K_max=8)
    if entropies:
        max_e = math.log(vocab_size)
        high_ent = [(e, k) for e, k in zip(entropies, k_vals) if e > 0.7 * max_e]
        low_ent  = [(e, k) for e, k in zip(entropies, k_vals) if e < 0.3 * max_e]
        avg_k_high = sum(k for _, k in high_ent) / len(high_ent) if high_ent else 0
        avg_k_low  = sum(k for _, k in low_ent)  / len(low_ent)  if low_ent  else 0
        avg_k_all  = sum(k_vals) / len(k_vals)
        print(f"  높은 엔트로피 (>70%) 토큰: avg K = {avg_k_high:.2f} (n={len(high_ent)})")
        print(f"  낮은 엔트로피 (<30%) 토큰: avg K = {avg_k_low:.2f} (n={len(low_ent)})")
        print(f"  전체 평균 K: {avg_k_all:.2f}")
        print(f"  K-effect: 높은 엔트로피 {avg_k_high:.2f} vs 낮은 엔트로피 {avg_k_low:.2f} "
              f"(차이: {avg_k_high - avg_k_low:.2f})")


if __name__ == "__main__":
    main()
