"""
PRISMCell — 에너지 경사하강 기반 상태 다이나믹스.

E(x) = ½‖u − Dx‖²_Π1 + ½‖(I−M)x‖²_Π2 + ½λ‖x‖²

내부시계 s: dx/ds = −∂E/∂x  (K번 반복)
외부시계 t: ΔM = Hebbian 갱신

Π1: 학습 가능 대각 (상수, 입력 독립)
    — CPU 속도 최적화: 입력 의존 MLP 제거
    — 이론: active inference의 모달리티 precision
Π2: 학습 가능 대각 (메모리 precision)

memory_mode: 'sliding' | 'full_M' | 'none'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


# ------------------------------------------------------------------ #
# 슬라이딩 윈도우 메모리                                               #
# ------------------------------------------------------------------ #

class SlidingMemory:
    """
    rank-r 슬라이딩 윈도우 연상기억.
    M x_q ≈ Σ_i w_i (k_i · x_q) v_i
    원형 버퍼 + mul+sum (einsum 없음, clone 최소화)
    """

    def __init__(self, d: int, rank: int, gamma: float):
        self.d = d
        self.rank = rank
        self.decay = 1.0 - gamma

    def init(self, B: int, device: torch.device) -> dict:
        return {
            "k_buf": torch.zeros(B, self.rank, self.d, device=device),
            "v_buf": torch.zeros(B, self.rank, self.d, device=device),
            "ptr"  : 0,
            "filled": 0,
        }

    def _weights(self, device: torch.device) -> torch.Tensor:
        r = self.rank
        return self.decay ** torch.arange(r, device=device).float()

    def apply(self, x_q: torch.Tensor, state: dict) -> torch.Tensor:
        if state["filled"] == 0:
            return x_q.new_zeros(x_q.shape)
        k_buf = state["k_buf"]
        v_buf = state["v_buf"]
        ptr   = state["ptr"]
        r     = self.rank
        # age: buf[ptr-1]=0, buf[ptr-2]=1, ...
        ages = torch.zeros(r, device=x_q.device)
        for i in range(r):
            ages[(ptr - 1 - i) % r] = float(i)
        w = (self.decay ** ages).unsqueeze(0)  # [1, r]
        dots = (k_buf * x_q.unsqueeze(1)).sum(-1) * w   # [B, r]
        return (dots.unsqueeze(-1) * v_buf).sum(1)       # [B, d]

    def apply_T(self, v_q: torch.Tensor, state: dict) -> torch.Tensor:
        if state["filled"] == 0:
            return v_q.new_zeros(v_q.shape)
        k_buf = state["k_buf"]
        v_buf = state["v_buf"]
        ptr   = state["ptr"]
        r     = self.rank
        ages = torch.zeros(r, device=v_q.device)
        for i in range(r):
            ages[(ptr - 1 - i) % r] = float(i)
        w = (self.decay ** ages).unsqueeze(0)
        dots = (v_buf * v_q.unsqueeze(1)).sum(-1) * w
        return (dots.unsqueeze(-1) * k_buf).sum(1)

    @torch.no_grad()
    def update(self, x_new: torch.Tensor, eps_new: torch.Tensor, state: dict) -> dict:
        ptr = state["ptr"]
        new_k = state["k_buf"].clone()
        new_v = state["v_buf"].clone()
        new_k[:, ptr, :] = x_new.detach()
        new_v[:, ptr, :] = eps_new.detach()
        return {
            "k_buf" : new_k,
            "v_buf" : new_v,
            "ptr"   : (ptr + 1) % self.rank,
            "filled": min(state["filled"] + 1, self.rank),
        }


# ------------------------------------------------------------------ #
# PRISMCell                                                           #
# ------------------------------------------------------------------ #

class PRISMCell(nn.Module):
    """
    단일 PRISM 셀.

    Args:
        d, emb_dim:      차원
        lam:             λ  (상태 정규화)
        K, alpha:        내부 반복, 스텝 크기
        mem_eta:         η  (Hebbian, full_M only)
        mem_gamma:       γ  (감쇠)
        memory_mode:     'sliding' | 'full_M' | 'none'
        mem_rank:        슬라이딩 rank
        approximate_grad: K-1 no_grad + 1 grad (실험적)
    """

    def __init__(
        self,
        d: int = 256,
        emb_dim: int = 64,
        lam: float = 0.01,
        K: int = 16,
        alpha: float = 0.05,
        mem_eta: float = 0.01,
        mem_gamma: float = 0.001,
        memory_mode: str = "sliding",
        mem_rank: int = 32,
        approximate_grad: bool = False,
    ):
        super().__init__()
        assert memory_mode in ("sliding", "full_M", "none")
        self.d = d
        self.emb_dim = emb_dim
        self.lam = lam
        self.K = K
        self.alpha = alpha
        self.mem_eta = mem_eta
        self.mem_gamma = mem_gamma
        self.memory_mode = memory_mode
        self.approximate_grad = approximate_grad

        if memory_mode == "sliding":
            self.sliding = SlidingMemory(d, mem_rank, mem_gamma)

        # 느린가중치 θ
        self.D = nn.Linear(d, emb_dim, bias=False)

        # Π1: 학습 가능 대각 precision (입력 독립, MLP 제거)
        self.log_pi1 = nn.Parameter(torch.zeros(emb_dim))

        # Π2: 기억 precision
        self.log_pi2 = nn.Parameter(torch.zeros(d))

        # 초기 상태 인코더
        self.x_init = nn.Linear(emb_dim, d)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.D.weight, std=0.02)
        nn.init.normal_(self.x_init.weight, std=0.02)
        nn.init.zeros_(self.x_init.bias)

    @property
    def pi1(self) -> torch.Tensor:
        return F.softplus(self.log_pi1)  # [emb_dim]

    @property
    def pi2(self) -> torch.Tensor:
        return F.softplus(self.log_pi2)  # [d]

    # ---------------------------------------------------------------- #
    # M 연산                                                            #
    # ---------------------------------------------------------------- #

    def _Mx(self, x: torch.Tensor, mem_state) -> torch.Tensor:
        if self.memory_mode == "none" or mem_state is None:
            return x.new_zeros(x.shape)
        if self.memory_mode == "sliding":
            return self.sliding.apply(x, mem_state)
        return torch.bmm(mem_state, x.unsqueeze(-1)).squeeze(-1)

    def _MtV(self, v: torch.Tensor, mem_state) -> torch.Tensor:
        if self.memory_mode == "none" or mem_state is None:
            return v.new_zeros(v.shape)
        if self.memory_mode == "sliding":
            return self.sliding.apply_T(v, mem_state)
        return torch.bmm(mem_state.transpose(-1, -2), v.unsqueeze(-1)).squeeze(-1)

    # ---------------------------------------------------------------- #
    # 에너지 + 그래디언트                                                #
    # ---------------------------------------------------------------- #

    def _neg_grad_E(self, x: torch.Tensor, u: torch.Tensor, mem_state) -> torch.Tensor:
        """
        −∂E/∂x = Dᵀ Π1 εin − (I−M)ᵀ Π2 εmem − λx
        Π1, Π2: 상수 대각행렬 (배치 독립)
        """
        # 지각 항
        eps_in  = u - self.D(x)           # [B, emb_dim]
        grad_in = (self.pi1 * eps_in) @ self.D.weight   # [B, d]

        # 기억 항
        if self.memory_mode != "none" and mem_state is not None:
            Mx       = self._Mx(x, mem_state)
            eps_mem  = x - Mx
            pi2_eps  = self.pi2 * eps_mem
            Mt_pi2e  = self._MtV(pi2_eps, mem_state)
            grad_mem = pi2_eps - Mt_pi2e        # (I−M)ᵀ Π2 εmem
        else:
            grad_mem = x.new_zeros(x.shape)

        # 정규화 항
        return grad_in - grad_mem - self.lam * x

    def energy(self, x: torch.Tensor, u: torch.Tensor, mem_state) -> torch.Tensor:
        eps_in = u - self.D(x)
        pi1    = self.pi1
        pi2    = self.pi2
        e_in   = 0.5 * (eps_in ** 2 * pi1).sum(-1)
        e_reg  = 0.5 * self.lam * (x ** 2).sum(-1)

        if self.memory_mode != "none" and mem_state is not None:
            Mx    = self._Mx(x, mem_state)
            eps_m = x - Mx
            e_mem = 0.5 * (eps_m ** 2 * pi2).sum(-1)
        else:
            e_mem = x.new_zeros(x.shape[0])

        return (e_in + e_mem + e_reg).mean()

    # ---------------------------------------------------------------- #
    # 내부시계 s                                                        #
    # ---------------------------------------------------------------- #

    def iterate(
        self,
        u: torch.Tensor,
        mem_state,
        x0: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
        return_energies: bool = False,
        training: bool = False,
    ):
        K = K if K is not None else self.K
        x = self.x_init(u) if x0 is None else x0

        if return_energies:
            energies: List[float] = []
            with torch.no_grad():
                for _ in range(K):
                    energies.append(self.energy(x, u, mem_state).item())
                    x = x + self.alpha * self._neg_grad_E(x, u, mem_state)
                energies.append(self.energy(x, u, mem_state).item())
            return x, energies

        if training and self.approximate_grad:
            # K-1 no_grad + 1 grad (근사 backward)
            with torch.no_grad():
                for _ in range(K - 1):
                    x = x + self.alpha * self._neg_grad_E(x, u, mem_state)
            x = x.detach()
            x = x + self.alpha * self._neg_grad_E(x, u, mem_state)
            return x

        # Full backprop (기본)
        with torch.set_grad_enabled(training):
            for _ in range(K):
                x = x + self.alpha * self._neg_grad_E(x, u, mem_state)
        return x

    # ---------------------------------------------------------------- #
    # 외부시계 t: Hebbian 기억 갱신                                     #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def update_memory(self, x: torch.Tensor, mem_state):
        if self.memory_mode == "none":
            return None
        if self.memory_mode == "sliding":
            Mx = self.sliding.apply(x, mem_state)
            eps_mem = x - Mx
            return self.sliding.update(x, eps_mem, mem_state)
        M = mem_state
        Mx = torch.bmm(M, x.unsqueeze(-1)).squeeze(-1)
        eps_mem = x - Mx
        dM = self.mem_eta * torch.bmm(
            eps_mem.unsqueeze(-1), x.unsqueeze(-2)
        ) - self.mem_gamma * M
        return M + dM

    # ---------------------------------------------------------------- #
    # 전체 PRISM 틱                                                     #
    # ---------------------------------------------------------------- #

    def forward(
        self,
        u: torch.Tensor,
        mem_state,
        x0: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, object]:
        x_star  = self.iterate(u, mem_state, x0, training=training)
        mem_new = self.update_memory(x_star.detach(), mem_state)
        return x_star, mem_new

    def init_state(self, B: int, device: torch.device):
        x = torch.zeros(B, self.d, device=device)
        if self.memory_mode == "none":
            mem = None
        elif self.memory_mode == "sliding":
            mem = self.sliding.init(B, device)
        else:
            mem = torch.zeros(B, self.d, self.d, device=device)
        return x, mem
