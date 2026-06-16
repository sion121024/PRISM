"""PRISM v1 vs PRISM-Base(v2) 비교: 파라미터 수, 수렴 속도, forward 속도."""

import time
import torch
from prism import PRISMCell, PRISMLM
from prism.energy_v2 import PRISMBaseCell
from prism.model_v2  import PRISMBase


def param_count(m): return sum(p.numel() for p in m.parameters())


def compare_convergence(d=128, B=16):
    print("=" * 60)
    print("에너지 수렴 비교 (같은 초기점, 다른 알고리즘)")
    print("=" * 60)

    torch.manual_seed(0)
    u  = torch.randn(B, d)
    x0 = torch.randn(B, d) * 0.1

    c1 = PRISMCell(d);     c1.D.weight.data.fill_(0); c1.D.weight.data.fill_diagonal_(0.5)
    c2 = PRISMBaseCell(d); c2.D.weight.data = c1.D.weight.data.clone()
    c2.log_pi1.data = c1.log_pi1.data.clone()
    c2.log_pi2.data = c1.log_pi2.data.clone()

    _, e1_k8  = c1.descend(x0.clone(), u, K=8,  step=0.05)
    _, e1_k20 = c1.descend(x0.clone(), u, K=20, step=0.05)
    _, e2_k6  = c2.descend(x0.clone(), u, K=6,  step=0.05, momentum=0.9)
    _, e2_k8  = c2.descend(x0.clone(), u, K=8,  step=0.05, momentum=0.9)

    print(f"v1 GD       K=8:   E[-1]={e1_k8[-1]:.4f}")
    print(f"v1 GD       K=20:  E[-1]={e1_k20[-1]:.4f}")
    print(f"v2 Nesterov K=6:   E[-1]={e2_k6[-1]:.4f}  (K 적게, 비슷한 수렴?)")
    print(f"v2 Nesterov K=8:   E[-1]={e2_k8[-1]:.4f}")


def compare_params(d=256, vocab=65):
    print()
    print("=" * 60)
    print("파라미터 수")
    print("=" * 60)
    v1 = PRISMLM(vocab, d=d, K=8)
    v2 = PRISMBase(vocab, d=d, K=6)
    p1, p2 = param_count(v1), param_count(v2)
    print(f"v1 (PRISMLM):    {p1:>10,}")
    print(f"v2 (PRISMBase):  {p2:>10,}")
    delta = p2 - p1
    print(f"  차이: {delta:+,}  ({'↑ 더 많음' if delta>0 else '↓ 더 적음'})")
    print()
    print("  v2 구성:")
    print(f"    embed+head:    {param_count(v2.embed)+param_count(v2.head):>8,}")
    print(f"    D (decoder):   {param_count(v2.cell.D):>8,}")
    print(f"    log_pi1+pi2:   {v2.cell.log_pi1.numel()*2:>8,}  (2d, 대각 Π)")
    print(f"    A+B (low-rank):{v2.cell.A.numel()+v2.cell.B.numel():>8,}  (2·d·r, r={v2.cell.rank})")


def compare_speed(d=128, B=32, T=32, vocab=65):
    print()
    print("=" * 60)
    print("forward 속도")
    print("=" * 60)
    tokens = torch.randint(0, vocab, (B, T))
    v1 = PRISMLM(vocab, d=d, K=8)
    v2 = PRISMBase(vocab, d=d, K=6)
    N = 3

    t0 = time.time()
    for _ in range(N): v1(tokens, use_deq=False)
    t1 = (time.time()-t0)/N

    t0 = time.time()
    for _ in range(N): v2(tokens)
    t2 = (time.time()-t0)/N

    print(f"v1 K=8  GD:         {t1:.3f}s")
    print(f"v2 K=6  Nesterov:   {t2:.3f}s")
    faster = 'v2' if t2 < t1 else 'v1'
    ratio  = max(t1,t2)/min(t1,t2)
    print(f"  {faster}가 {ratio:.1f}x 빠름")


if __name__ == "__main__":
    compare_convergence()
    compare_params()
    compare_speed()
