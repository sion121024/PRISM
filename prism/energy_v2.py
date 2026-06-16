"""
PRISM-Base: balanced energy-descent cell.

Three principled additions over v1 — each minimal:

1. Nesterov momentum on inner descent
   Anderson보다 단순, lstsq 오버헤드 없음, 수렴 속도 유사.
   이론: heavy-ball / Nesterov = O(1/k²) vs GD O(1/k).

2. Learned diagonal precision Π (벡터, 네트워크 아님)
   파라미터 2d 추가 (d << d²). 입력 의존은 포기, 학습된 channel-wise 선택성 유지.

3. Low-rank fast weight M = A·Bᵀ  (r = d//4)
   메모리 O(d²) → O(dr). 파라미터도 줄어듦.

추가 안 한 것: 입력의존 Π 네트워크, 슬롯, attention — 단순함 우선.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PRISMBaseCell(nn.Module):
    def __init__(self, d: int, rank: int | None = None, lam: float = 0.01):
        super().__init__()
        self.d    = d
        self.rank = rank or max(1, d // 4)
        self.lam  = lam

        # Slow weights θ
        self.D = nn.Linear(d, d, bias=False)

        # Learned diagonal precision (2d params vs d² for a network)
        self.log_pi1 = nn.Parameter(torch.zeros(d))   # perception channel
        self.log_pi2 = nn.Parameter(torch.zeros(d))   # memory channel

        # Low-rank fast weight M ≈ A · Bᵀ
        self.register_buffer("A", torch.zeros(d, self.rank))
        self.register_buffer("B", torch.zeros(d, self.rank))

        nn.init.orthogonal_(self.D.weight, gain=0.5)

    # ------------------------------------------------------------------ #
    #  Low-rank M operations                                              #
    # ------------------------------------------------------------------ #

    def _Mx(self, x: torch.Tensor) -> torch.Tensor:
        """M·x = A·(Bᵀ·x)  in O(Br·d) instead of O(Bd²)."""
        return (x @ self.B) @ self.A.T      # (B, d)

    # ------------------------------------------------------------------ #
    #  Energy and gradient                                                #
    # ------------------------------------------------------------------ #

    def energy(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pi1 = self.log_pi1.exp()
        pi2 = self.log_pi2.exp()
        eps_in  = u - x @ self.D.weight.T
        eps_mem = x - self._Mx(x)
        return (0.5 * (eps_in**2  * pi1).sum(-1)
              + 0.5 * (eps_mem**2 * pi2).sum(-1)
              + 0.5 * self.lam * (x**2).sum(-1))

    def grad_E(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pi1 = self.log_pi1.exp()
        pi2 = self.log_pi2.exp()
        eps_in  = u - x @ self.D.weight.T
        eps_mem = x - self._Mx(x)
        # ∂eps_mem/∂x = (I - Mᵀ),  Mᵀ·v = B·(Aᵀ·v)
        d_mem = eps_mem - (eps_mem @ self.A) @ self.B.T
        return (-(eps_in * pi1) @ self.D.weight
                + d_mem * pi2
                + self.lam * x)

    # ------------------------------------------------------------------ #
    #  Nesterov-accelerated inner descent                                 #
    # ------------------------------------------------------------------ #

    def descend(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        K: int = 6,
        step: float = 0.1,
        momentum: float = 0.9,
        adaptive: bool = False,
        tol: float = 1e-4,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Nesterov momentum: O(1/k²) vs plain GD O(1/k).
        Same K, faster convergence. No extra buffers beyond one prev iterate.
        """
        v = x.clone()       # velocity / momentum state
        energies = []

        for k in range(K):
            # Nesterov look-ahead
            x_look = x + momentum * (x - v) if k > 0 else x
            g = self.grad_E(x_look, u)
            v_new = x
            x     = x_look - step * g
            v     = v_new

            E = self.energy(x, u).mean().item()
            energies.append(E)
            if adaptive and len(energies) > 1 and abs(energies[-1] - energies[-2]) < tol:
                break

        return x, energies

    # ------------------------------------------------------------------ #
    #  Fast weight update                                                 #
    # ------------------------------------------------------------------ #

    def update_M(self, x: torch.Tensor, eta: float = 0.01, gamma: float = 0.999):
        with torch.no_grad():
            eps_mem = x - self._Mx(x)
            xm = x.mean(0)
            em = eps_mem.mean(0)
            self.A.mul_(1 - eta * gamma)
            self.B.mul_(1 - eta * gamma)
            self.A.add_(eta * em.unsqueeze(1) @ (self.B.T @ xm).unsqueeze(0))
            self.B.add_(eta * xm.unsqueeze(1) @ (self.A.T @ em).unsqueeze(0))

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        K: int = 6,
        step: float = 0.1,
        momentum: float = 0.9,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, list[float]]:
        x, energies = self.descend(x, u, K=K, step=step, momentum=momentum)
        if update_memory:
            self.update_M(x)
        return x, energies

    def reset_memory(self):
        self.A.zero_()
        self.B.zero_()
