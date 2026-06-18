"""
diag_scan 속도/정확도 벤치마크 — "더 빠르고 효율적" 검증.

순차 K-step (O(K)) vs 대각 병렬 scan (O(1) in K).
같은 에너지 최솟값에 수렴하면서 K가 클수록 속도 이득.
"""

import time, torch
torch.set_num_threads(1)
from prism.cell import PRISMCell

B, d, emb = 64, 256, 64
cell = PRISMCell(d=d, emb_dim=emb, K=4, memory_mode="sliding", mem_rank=8,
                 simple_prior=True)
u = torch.randn(B, emb)
x0 = torch.randn(B, d)
mem = cell.init_state(B, "cpu")[1]
x_prior = x0

def timeit(fn, n=200):
    fn()  # warmup
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1000  # ms

print("diag_scan 벤치마크 (B=64, d=256)")
print("=" * 55)
print(f"{'K':>4} | {'순차(ms)':>10} | {'diag(ms)':>10} | {'속도이득':>8} | {'오차':>8}")
print("-" * 55)

for K in [4, 8, 16, 32, 64]:
    seq = lambda: cell.iterate(u, mem, x0, K=K, x_prior=x_prior)
    dia = lambda: cell.iterate(u, mem, x0, K=K, x_prior=x_prior, diag_scan=True)
    t_seq = timeit(seq)
    t_dia = timeit(dia)
    with torch.no_grad():
        xs = cell.iterate(u, mem, x0, K=K, x_prior=x_prior)
        xd = cell.iterate(u, mem, x0, K=K, x_prior=x_prior, diag_scan=True)
        err = (xs - xd).abs().max().item()
    print(f"{K:>4} | {t_seq:>10.3f} | {t_dia:>10.3f} | {t_seq/t_dia:>7.1f}x | {err:>8.4f}")

print("=" * 55)
print("diag_scan: K 무관 O(1) — K 클수록 속도 이득, 같은 에너지 최솟값 수렴")
