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
    설계 원본 Hebbian ΔM = η(εmem ⊗ x) − γM 의 저랭크 근사.

    M x_q   = Σ_i w_i (x̂_i · x_q) ê_i      [apply   — 키=x̂, 값=ê]
    Mᵀ v_q  = Σ_i w_i (ê_i · v_q) x̂_i       [apply_T — 키=ê, 값=x̂]

    x̂ = normalize(x),  ê = normalize(εmem)
    이전 대칭 Hopfield와 달리 비대칭(key≠value) → 오류신호 기반 연상기억.

    상태: (x_buf [B,r,d], e_buf [B,r,d])  ← 순수 텐서, torch.compile 호환
    slot 0 = newest, slot rank-1 = oldest
    """

    def __init__(self, d: int, rank: int, gamma: float, scale: float = 1.0):
        self.d = d
        self.rank = rank
        self.decay = 1.0 - gamma
        self.scale = scale / rank  # spectral_norm(M) ≈ scale

    def _w(self, device: torch.device) -> torch.Tensor:
        return (self.decay ** torch.arange(self.rank, device=device).float()).unsqueeze(0)

    def init(self, B: int, device: torch.device):
        z = torch.zeros(B, self.rank, self.d, device=device)
        return (z, z.clone())  # (x_buf, e_buf)

    def apply(self, x_q: torch.Tensor, state) -> torch.Tensor:
        """M x_q = Σ_i w_i (x̂_i · x_q) ê_i"""
        x_buf, e_buf = state
        dots = (x_buf * x_q.unsqueeze(1)).sum(-1) * self._w(x_q.device) * self.scale
        return (dots.unsqueeze(-1) * e_buf).sum(1)

    def apply_T(self, v_q: torch.Tensor, state) -> torch.Tensor:
        """Mᵀ v_q = Σ_i w_i (ê_i · v_q) x̂_i"""
        x_buf, e_buf = state
        dots = (e_buf * v_q.unsqueeze(1)).sum(-1) * self._w(v_q.device) * self.scale
        return (dots.unsqueeze(-1) * x_buf).sum(1)

    @torch.no_grad()
    def update(self, x_new: torch.Tensor, eps_new: torch.Tensor, state) -> tuple:
        """(x̂_new, ê_new) 쌍을 slot 0에 삽입, 이전 항목 한 칸 밀어냄."""
        x_buf, e_buf = state
        xk = x_new.detach()
        xk = xk / xk.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ek = eps_new.detach()
        ek = ek / ek.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        new_x = torch.empty_like(x_buf)
        new_e = torch.empty_like(e_buf)
        new_x[:, 0, :] = xk
        new_x[:, 1:, :] = x_buf[:, :-1, :]
        new_e[:, 0, :] = ek
        new_e[:, 1:, :] = e_buf[:, :-1, :]
        return (new_x, new_e)


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
        decoder: str = "linear",
        dec_hidden: int = 128,
        state_norm: Optional[bool] = None,
        mem_scale: float = 1.0,
        use_prior: bool = False,
        simple_prior: bool = False,  # True → μ = x_prev (파라미터 없는 identity prior)
    ):
        super().__init__()
        assert memory_mode in ("sliding", "full_M", "none")
        assert decoder in ("linear", "mlp")
        if state_norm is None:
            state_norm = False  # K-루프 내 정규화 없음이 기본 (에너지 단조감소 보장)
        self.state_norm = state_norm
        self.d = d
        self.emb_dim = emb_dim
        self.lam = lam
        self.K = K
        self.alpha = alpha
        self.mem_eta = mem_eta
        self.mem_gamma = mem_gamma
        self.memory_mode = memory_mode
        self.approximate_grad = approximate_grad
        self.decoder = decoder
        self.dec_hidden = dec_hidden
        self.use_prior = use_prior

        if memory_mode == "sliding":
            self.sliding = SlidingMemory(d, mem_rank, mem_gamma, scale=mem_scale)

        # 느린가중치 θ — 생성모델 g: ℝ^d → ℝ^emb
        #   linear: g(x) = D x          → E 가 2차식 (사고가 자명)
        #   mlp   : g(x) = W2·tanh(W1 x) → E 가 비볼록 (진짜 사고)
        if decoder == "linear":
            self.D = nn.Linear(d, emb_dim, bias=False)
        else:
            self.dec_W1 = nn.Parameter(torch.empty(dec_hidden, d))
            self.dec_b1 = nn.Parameter(torch.zeros(dec_hidden))
            self.dec_W2 = nn.Parameter(torch.empty(emb_dim, dec_hidden))
            self.dec_b2 = nn.Parameter(torch.zeros(emb_dim))

        # Π1: 학습 가능 대각 precision (입력 독립)
        self.log_pi1 = nn.Parameter(torch.zeros(emb_dim))

        # Π2: 기억 precision
        self.log_pi2 = nn.Parameter(torch.zeros(d))

        # Prior 항: ½‖x − μ(x_prev)‖²_Π3
        # simple_prior=True: μ = x_prev (identity, 파라미터 없음) ← 단순하고 빠름
        # use_prior=True:    μ = MLP(x_prev) (학습, +18K params)  ← 더 표현력 있음
        self.simple_prior = simple_prior
        if use_prior and not simple_prior:
            self.prior_mu = nn.Sequential(
                nn.Linear(d, d // 4),
                nn.GELU(),
                nn.Linear(d // 4, d),
            )
        if use_prior or simple_prior:
            self.log_pi3 = nn.Parameter(torch.zeros(d))

        # 초기 상태 인코더
        self.x_init = nn.Linear(emb_dim, d)

        if state_norm:
            self.norm_gain = nn.Parameter(torch.ones(d))

        self._init_weights()

    def _init_weights(self):
        if self.decoder == "linear":
            nn.init.normal_(self.D.weight, std=0.02)
        else:
            nn.init.normal_(self.dec_W1, std=0.02)
            nn.init.normal_(self.dec_W2, std=0.02)
        nn.init.normal_(self.x_init.weight, std=0.02)
        nn.init.zeros_(self.x_init.bias)

    # ---------------------------------------------------------------- #
    # 생성모델 g 와 그 Jacobianᵀ                                        #
    # ---------------------------------------------------------------- #

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """g(x): 상태 → 예측된 입력 (ℝ^d → ℝ^emb)."""
        if self.decoder == "linear":
            return self.D(x)
        h = torch.tanh(x @ self.dec_W1.t() + self.dec_b1)   # [B, dec_hidden]
        return h @ self.dec_W2.t() + self.dec_b2            # [B, emb]

    def _dec_grad(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Jacobian(g)ᵀ · w   (w ∈ ℝ^emb → 결과 ℝ^d).
        −∂/∂x ½‖u−g(x)‖²_Π1 = J_gᵀ (Π1 (u−g(x))) 계산에 사용.
        """
        if self.decoder == "linear":
            return w @ self.D.weight                        # Dᵀ w
        # g = W2·tanh(W1 x + b1) + b2  →  J_gᵀ w = W1ᵀ (tanh'(z) ⊙ (W2ᵀ w))
        z = x @ self.dec_W1.t() + self.dec_b1
        h = torch.tanh(z)
        b = w @ self.dec_W2                                  # W2ᵀ w   [B, dec_hidden]
        c = b * (1.0 - h * h)                                # tanh'(z) ⊙ ·
        return c @ self.dec_W1                               # W1ᵀ c   [B, d]

    @property
    def pi1(self) -> torch.Tensor:
        return F.softplus(self.log_pi1)  # [emb_dim]

    @property
    def pi2(self) -> torch.Tensor:
        return F.softplus(self.log_pi2)  # [d]

    @property
    def pi3(self) -> torch.Tensor:
        return F.softplus(self.log_pi3) if (self.use_prior or self.simple_prior) else None  # [d]

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

    def _neg_grad_E(
        self, x: torch.Tensor, u: torch.Tensor, mem_state,
        x_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        −∂E/∂x = Dᵀ Π1 εin − (I−M)ᵀ Π2 εmem − Π3(x−μ) − λx
        μ = prior_mu(x_prev): 에너지 내부 상태 전이 prior (optional)
        """
        # 지각 항:  J_gᵀ Π1 (u − g(x))
        eps_in  = u - self._decode(x)
        grad_in = self._dec_grad(x, self.pi1 * eps_in)

        # 기억 항
        if self.memory_mode != "none" and mem_state is not None:
            Mx       = self._Mx(x, mem_state)
            eps_mem  = x - Mx
            pi2_eps  = self.pi2 * eps_mem
            Mt_pi2e  = self._MtV(pi2_eps, mem_state)
            grad_mem = pi2_eps - Mt_pi2e
        else:
            grad_mem = x.new_zeros(x.shape)

        # Prior 항: −Π3(x − μ)
        # simple_prior: μ = x_prior = x_prev (identity, 파라미터 없음)
        # use_prior:    μ = prior_mu(x_prev) (MLP)
        if (self.use_prior or self.simple_prior) and x_prior is not None:
            grad_prior = self.pi3 * (x_prior - x)
        else:
            grad_prior = x.new_zeros(x.shape)

        return grad_in - grad_mem + grad_prior - self.lam * x

    def energy(
        self, x: torch.Tensor, u: torch.Tensor, mem_state,
        x_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps_in = u - self._decode(x)
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

        if (self.use_prior or self.simple_prior) and x_prior is not None:
            eps_p = x - x_prior
            e_prior = 0.5 * (eps_p ** 2 * self.pi3).sum(-1)
        else:
            e_prior = x.new_zeros(x.shape[0])

        return (e_in + e_mem + e_prior + e_reg).mean()

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
        adaptive: bool = False,
        K_min: int = 1,
        K_tol: float = 1e-4,
        x_prior: Optional[torch.Tensor] = None,
    ):
        """
        내부시계 s: K번 에너지 하강.

        x_prior: prior_mu(x_prev) — ½‖x−μ‖²_Π3 항으로 에너지 내부 상태 전이.
        adaptive=True (추론 전용): 에너지 수렴 시 조기 종료 (이중시계 설계).
        """
        K = K if K is not None else self.K
        x = self.x_init(u) if x0 is None else x0

        if return_energies:
            energies: List[float] = []
            with torch.no_grad():
                for _ in range(K):
                    energies.append(self.energy(x, u, mem_state, x_prior).item())
                    x = x + self.alpha * self._neg_grad_E(x, u, mem_state, x_prior)
                energies.append(self.energy(x, u, mem_state, x_prior).item())
            return x, energies

        if training and self.approximate_grad:
            with torch.no_grad():
                for _ in range(K - 1):
                    x = x + self.alpha * self._neg_grad_E(x, u, mem_state, x_prior)
            x = x.detach()
            x = x + self.alpha * self._neg_grad_E(x, u, mem_state, x_prior)
            return x

        # 적응형 K (추론 전용): 에너지 수렴 시 조기 종료
        if adaptive and not training:
            with torch.no_grad():
                prev_e = float('inf')
                for k in range(K):
                    e = self.energy(x, u, mem_state, x_prior).item()
                    if k >= K_min and abs(prev_e - e) / (abs(prev_e) + 1e-8) < K_tol:
                        break
                    prev_e = e
                    x = x + self.alpha * self._neg_grad_E(x, u, mem_state, x_prior)
            return x

        # Full backprop (기본) — raw x 공간에서 정확한 에너지 경사하강
        with torch.set_grad_enabled(training):
            for _ in range(K):
                x = x + self.alpha * self._neg_grad_E(x, u, mem_state, x_prior)
        return x

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-norm RMS 정규화 — 에너지/디코더가 '읽는' 유계 상태.
        raw x 는 루프에서 누적(반복+재귀 보존)되고, g·M 은 항상 x̂=_rms(x) 를
        봐서 안정. transformer pre-norm 과 동일한 발상.
        linear 디코더에선 no-op (state_norm=False) — Stage 1/2 동작 보존.
        """
        if not self.state_norm:
            return x
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        return x * rms * self.norm_gain

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
        x_prior: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
    ) -> Tuple[torch.Tensor, object]:
        x_star  = self.iterate(u, mem_state, x0, K=K, training=training, x_prior=x_prior)
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
