"""
Stage 1 사활 검증: 에너지 수렴 확인.

"에너지가 내부 반복에 따라 단조감소하는가?"

실행:
  python verify_convergence.py
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prism import PRISMLangModel


def verify_energy_convergence(
    d: int = 128,
    emb_dim: int = 32,
    K: int = 32,
    alpha: float = 0.05,
    batch_size: int = 8,
    vocab_size: int = 64,
    n_trials: int = 5,
    seed: int = 42,
    memory_mode: str = "sliding",
    mem_rank: int = 4,
):
    """K번 반복에 걸쳐 에너지가 감소하는지 확인."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=emb_dim,
        K=K, alpha=alpha, memory_mode=memory_mode, mem_rank=mem_rank,
    ).to(device)
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    all_decreasing = True
    print(f"{'Trial':>5} | {'E[0]':>10} | {'E[K]':>10} | {'단조감소?':>8}")
    print("-" * 45)

    for trial in range(n_trials):
        tokens = torch.randint(0, vocab_size, (batch_size,), device=device)
        u = model.embed(tokens)
        _, mem = model.cell.init_state(batch_size, device)

        _, energies = model.cell.iterate(u, mem, return_energies=True, K=K)

        monotone = all(
            energies[i] >= energies[i + 1] - 1e-6
            for i in range(len(energies) - 1)
        )
        if not monotone:
            all_decreasing = False

        E0, EK = energies[0], energies[-1]
        print(
            f"{trial + 1:>5} | {E0:>10.4f} | {EK:>10.4f}"
            f" | {'YES ✓' if monotone else 'NO  ✗':>8}"
        )
        axes[0].plot(energies, label=f"trial {trial + 1}", alpha=0.7)

    axes[0].set_xlabel("내부 반복 s")
    axes[0].set_ylabel("에너지 E(x)")
    axes[0].set_title("에너지 궤적 (K-step 경사하강)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    tokens = torch.randint(0, vocab_size, (batch_size,), device=device)
    u = model.embed(tokens)
    _, mem = model.cell.init_state(batch_size, device)
    _, energies = model.cell.iterate(u, mem, return_energies=True, K=K)
    E0 = energies[0]
    normalized = [(e - energies[-1]) / (E0 - energies[-1] + 1e-8)
                  for e in energies]
    axes[1].semilogy([1 - n + 1e-8 for n in normalized], 'b-o', markersize=3)
    axes[1].set_xlabel("내부 반복 s")
    axes[1].set_ylabel("(E - E*) / (E0 - E*)")
    axes[1].set_title("수렴 속도 (log scale)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("energy_convergence.png", dpi=150)
    print(f"\n그래프 저장: energy_convergence.png")
    print(f"\n결론: {'에너지 단조감소 확인 ✓' if all_decreasing else '에너지 수렴 실패 ✗ — alpha 줄이기 필요'}")

    return all_decreasing


def verify_fixed_point(
    d: int = 128,
    emb_dim: int = 32,
    K: int = 64,
    alpha: float = 0.05,
    tol: float = 1e-4,
    batch_size: int = 8,
    vocab_size: int = 64,
    memory_mode: str = "none",
):
    """K 증가에 따라 상태가 고정점에 수렴하는지 확인."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISMLangModel(
        vocab_size=vocab_size, d=d, emb_dim=emb_dim, K=1, alpha=alpha,
        memory_mode=memory_mode,
    ).to(device)
    model.eval()

    tokens = torch.randint(0, vocab_size, (batch_size,), device=device)
    u = model.embed(tokens)
    _, mem = model.cell.init_state(batch_size, device)

    x = model.cell.x_init(u)
    residuals = []
    with torch.no_grad():
        for k in range(K):
            x_new = x + alpha * model.cell._neg_grad_E(x, u, mem)
            residual = (x_new - x).norm(dim=-1).mean().item()
            residuals.append(residual)
            x = x_new

    conv_step = next(
        (k for k, r in enumerate(residuals) if r < tol), K
    )
    print(f"\n고정점 수렴: tol={tol} 도달 스텝 = {conv_step}/{K}")
    print(f"최종 잔차: {residuals[-1]:.2e}")

    return residuals


if __name__ == "__main__":
    torch.set_num_threads(1)
    print("=" * 50)
    print("PRISM Stage 1 — 에너지 수렴 검증")
    print("=" * 50)

    ok = verify_energy_convergence()
    residuals = verify_fixed_point()

    print("\n검증 완료.")
    if not ok:
        print("⚠  alpha를 줄이거나 모델 구조를 점검하세요.")
