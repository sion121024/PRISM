"""
적응형 K(t) 검증: "어려운 토큰 = 더 많은 K-step"

학습된 PRISM 모델을 로드해서:
  1. 고정 K vs 적응형 K의 ppl 비교
  2. 토큰별 실제 사용된 K 분포
  3. 예측 오류(loss)가 높은 토큰에서 K가 더 많이 사용되는지 확인

사용:
  python verify_adaptive_k.py --model_path model.pt --k_max 8
  (모델 없으면 빠른 학습 후 검증)
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F
from collections import defaultdict

torch.set_num_threads(1)

from prism import PRISMLangModel
from tasks import TinyShakespeare


def measure_adaptive_k(model, val_loader, device, K_max=8, K_tol=1e-4, K_min=1):
    """토큰별 적응형 K 사용량과 loss 측정."""
    model.eval()
    cell = model.cell

    k_used_list = []
    loss_list = []
    energy_drop_list = []

    with torch.no_grad():
        for tokens in val_loader:
            tokens = tokens.to(device)
            B, T = tokens.shape
            x, mem = model.init_state(B, device)
            u_all = model.embed(tokens[:, :-1])

            for t in range(T - 1):
                u_raw = u_all[:, t]
                if model.use_urec:
                    u = model.u_rec2(F.gelu(model.u_rec1(
                        torch.cat([u_raw, x], dim=-1))))
                else:
                    u = u_raw

                if model.simple_prior:
                    x_prior = x
                elif model.use_prior:
                    x_prior = cell.prior_mu(x)
                else:
                    x_prior = None

                # 적응형 K: 에너지 수렴 추적
                prev_e = float('inf')
                k_stop = K_max
                x_t = x.clone()
                for k in range(K_max):
                    e = cell.energy(x_t, u, mem, x_prior).item()
                    if k >= K_min and abs(prev_e - e) / (abs(prev_e) + 1e-8) < K_tol:
                        k_stop = k
                        break
                    prev_e = e
                    x_t = x_t + cell.alpha * cell._neg_grad_E(x_t, u, mem, x_prior)

                # 예측 loss
                logit = model.output_proj(x_t)
                target = tokens[:, t + 1]
                loss_t = F.cross_entropy(logit, target, reduction='none')

                k_used_list.append(k_stop)
                loss_list.extend(loss_t.cpu().tolist())

                # 메모리 갱신
                mem = cell.update_memory(x_t.detach(), mem)
                rms = x_t.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
                x = x_t * rms

            # 배치 1개만 측정 (속도)
            break

    return k_used_list, loss_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--K_max", type=int, default=8)
    p.add_argument("--K_tol", type=float, default=1e-4)
    p.add_argument("--train_epochs", type=int, default=3)
    p.add_argument("--d", type=int, default=176)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    device = torch.device(args.device)

    train_ds = TinyShakespeare(block_size=args.block_size, split="train")
    val_ds   = TinyShakespeare(block_size=args.block_size, split="val")
    vocab_size = train_ds.vocab_size
    train_loader = train_ds.get_loader(batch_size=args.batch_size)
    val_loader   = val_ds.get_loader(batch_size=4, shuffle=False)

    print(f"모델 학습 중 ({args.train_epochs} epochs)...")
    model = PRISMLangModel(
        vocab_size=vocab_size, d=args.d, emb_dim=args.emb_dim,
        K=args.K_max, alpha=args.alpha, memory_mode="sliding",
        mem_rank=24, mem_scale=4.0,
        decoder="mlp", dec_hidden=max(16, args.d // 4),
        simple_prior=True, use_urec=True,
    ).to(device)
    print(f"  params: {model.num_params():,}")

    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.train_epochs * len(train_loader))

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        total = 0.0
        for tokens in train_loader:
            tokens = tokens.to(device)
            out = model(tokens)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += out["loss"].item()
        print(f"  epoch {epoch}: train_loss {total/len(train_loader):.4f}")

    print(f"\n적응형 K 분석 (K_max={args.K_max}, K_tol={args.K_tol})...")
    t0 = time.time()
    k_used, losses = measure_adaptive_k(
        model, val_loader, device, args.K_max, args.K_tol)
    print(f"  측정 완료 ({time.time()-t0:.1f}s)")

    k_arr = torch.tensor(k_used, dtype=torch.float)
    l_arr = torch.tensor(losses[:len(k_used)])

    print(f"\n  K 사용 분포 (0~{args.K_max}회):")
    for k in range(args.K_max + 1):
        cnt = (k_arr == k).sum().item()
        bar = "#" * int(cnt * 40 / len(k_arr))
        print(f"    K={k:2d}: {cnt:4d}회 ({cnt/len(k_arr)*100:4.1f}%) {bar}")

    print(f"\n  평균 K 사용: {k_arr.mean():.2f} / {args.K_max}")

    # 상위 25% 고손실 vs 하위 25% 저손실 토큰의 K 비교
    q75 = torch.quantile(l_arr, 0.75)
    q25 = torch.quantile(l_arr, 0.25)
    high_loss_k = k_arr[l_arr > q75].mean().item()
    low_loss_k  = k_arr[l_arr < q25].mean().item()
    print(f"\n  어려운 토큰 (상위 25% loss) 평균 K: {high_loss_k:.2f}")
    print(f"  쉬운 토큰 (하위 25% loss) 평균 K:   {low_loss_k:.2f}")
    if high_loss_k > low_loss_k:
        print(f"  ✓ 어려운 토큰이 더 많은 K 사용 (차이 {high_loss_k-low_loss_k:.2f})")
    else:
        print(f"  ✗ K 분화 없음 (K_tol 조정 필요)")


if __name__ == "__main__":
    main()
