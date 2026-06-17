"""
PRISM 멀티모달 확장.

설계 원칙 — 에너지에 시각 항 추가:

  E(x) = ½‖ũ_text − g(x)‖²_Π1      ← 텍스트 지각  (기존)
       + ½‖v_vis − g_vis(x)‖²_Π_vis ← 시각 지각   (신규)
       + ½‖(I−M)x‖²_Π2              ← 기억        (기존)
       + ½‖x − μ(x_prev)‖²_Π3      ← Prior       (기존)
       + ½λ‖x‖²                     ← 정규화       (기존)

시각 인코더: 이미지 패치 → v_vis ∈ ℝ^vis_dim
시각 디코더: g_vis(x) = W_vis · x (선형, 또는 MLP)

추가 파라미터:
  - PatchEmbed: Conv2d (patch 분할 + 투영)
  - vis_dec_W: ℝ^{vis_dim × d} (시각 디코더)
  - log_pi_vis: ℝ^vis_dim (시각 precision)

K-step 에너지 하강이 텍스트+시각 동시 처리 — 별도 fusion layer 불필요.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .cell import PRISMCell


class PatchEmbed(nn.Module):
    """이미지 → 패치 시퀀스 → 시각 특징 벡터."""

    def __init__(self, img_size: int = 28, patch_size: int = 7,
                 in_channels: int = 1, vis_dim: int = 64):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, vis_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 패치 평균 → 단일 벡터

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: [B, C, H, W] → [B, vis_dim]"""
        x = self.proj(img)       # [B, vis_dim, H/p, W/p]
        x = self.pool(x)         # [B, vis_dim, 1, 1]
        return x.flatten(1)      # [B, vis_dim]


class PRISMMultimodalCell(PRISMCell):
    """
    PRISM 셀 + 시각 지각 항.

    에너지:
      E(x) = E_text(x) + ½‖v − g_vis(x)‖²_Π_vis + E_mem(x) + E_prior(x) + E_reg(x)

    텍스트 없는 토큰(v=None): E_vis 항 = 0 (자연스럽게 텍스트 전용 처리)
    시각 없는 토큰(u=None): E_text 항 = 0 (자연스럽게 시각 전용 처리)
    """

    def __init__(self, vis_dim: int = 64, vis_dec: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.vis_dim = vis_dim

        # 시각 디코더 g_vis: ℝ^d → ℝ^vis_dim
        if vis_dec == "linear":
            self.vis_dec_W = nn.Parameter(torch.empty(vis_dim, self.d))
            nn.init.normal_(self.vis_dec_W, std=0.02)
        else:
            self.vis_dec_W1 = nn.Parameter(torch.empty(vis_dim, self.d))
            self.vis_dec_b1 = nn.Parameter(torch.zeros(vis_dim))
            self.vis_dec_W2 = nn.Parameter(torch.empty(vis_dim, vis_dim))
            nn.init.normal_(self.vis_dec_W1, std=0.02)
            nn.init.normal_(self.vis_dec_W2, std=0.02)
        self.vis_dec = vis_dec

        # 시각 precision Π_vis (학습 가능 대각)
        self.log_pi_vis = nn.Parameter(torch.zeros(vis_dim))

    @property
    def pi_vis(self) -> torch.Tensor:
        return F.softplus(self.log_pi_vis)

    def _decode_vis(self, x: torch.Tensor) -> torch.Tensor:
        """g_vis(x): 상태 → 예측된 시각 특징."""
        if self.vis_dec == "linear":
            return x @ self.vis_dec_W.t()
        h = torch.tanh(x @ self.vis_dec_W1.t() + self.vis_dec_b1)
        return h @ self.vis_dec_W2.t()

    def _dec_grad_vis(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """J_{g_vis}ᵀ · w"""
        if self.vis_dec == "linear":
            return w @ self.vis_dec_W
        h = torch.tanh(x @ self.vis_dec_W1.t() + self.vis_dec_b1)
        b = w @ self.vis_dec_W2
        c = b * (1.0 - h * h)
        return c @ self.vis_dec_W1

    def _neg_grad_E(
        self, x: torch.Tensor, u: torch.Tensor, mem_state,
        x_prior: Optional[torch.Tensor] = None,
        v_vis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        −∂E/∂x = (텍스트 항) + (시각 항) + (기억 항) + (prior 항) − λx
        v_vis: 시각 특징 [B, vis_dim], None이면 시각 항 = 0
        """
        # 텍스트 지각
        eps_in  = u - self._decode(x)
        grad_in = self._dec_grad(x, self.pi1 * eps_in)

        # 시각 지각 (멀티모달 신규)
        if v_vis is not None:
            eps_vis  = v_vis - self._decode_vis(x)
            grad_vis = self._dec_grad_vis(x, self.pi_vis * eps_vis)
        else:
            grad_vis = x.new_zeros(x.shape)

        # 기억 항
        if self.memory_mode != "none" and mem_state is not None:
            Mx      = self._Mx(x, mem_state)
            eps_mem = x - Mx
            pi2_eps = self.pi2 * eps_mem
            grad_mem = pi2_eps - self._MtV(pi2_eps, mem_state)
        else:
            grad_mem = x.new_zeros(x.shape)

        # Prior 항
        if self.use_prior and x_prior is not None:
            grad_prior = self.pi3 * (x_prior - x)
        else:
            grad_prior = x.new_zeros(x.shape)

        return grad_in + grad_vis - grad_mem + grad_prior - self.lam * x

    def iterate(
        self, u: torch.Tensor, mem_state, x0=None,
        K=None, training: bool = False,
        x_prior=None, v_vis=None, **kwargs,
    ):
        """K-step 에너지 하강 (텍스트 + 시각 동시)."""
        K = K if K is not None else self.K
        x = self.x_init(u) if x0 is None else x0

        with torch.set_grad_enabled(training):
            for _ in range(K):
                x = x + self.alpha * self._neg_grad_E(
                    self._rms(x), u, mem_state,
                    x_prior=x_prior, v_vis=v_vis,
                )
        return self._rms(x)

    def forward(self, u, mem_state, x0=None, training=True,
                x_prior=None, K=None, v_vis=None):
        x_star  = self.iterate(u, mem_state, x0, K=K, training=training,
                               x_prior=x_prior, v_vis=v_vis)
        mem_new = self.update_memory(x_star.detach(), mem_state)
        return x_star, mem_new


class PRISMMultimodalModel(nn.Module):
    """
    PRISM 멀티모달 언어 모델.

    입력:
      - tokens [B, T]: 텍스트 토큰 시퀀스
      - images [B, T, C, H, W]: 각 토큰에 대응하는 이미지 (없으면 None)
        (이미지 없는 토큰: images[:, t] = None 또는 mask 사용)

    처리 흐름 (토큰 t마다):
      1. u_t = Embed(s_t) + u_rec(u_t, x_prev)
      2. v_t = PatchEmbed(img_t)  [이미지 있을 때]
      3. x_t* = iterate(u_t, mem, x_prev; v_vis=v_t)  [K회 에너지 하강]
      4. mem_t = Hebbian(x_t*)
      5. logits_t = Unembed(x_t*)
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 256,
        emb_dim: int = 64,
        K: int = 4,
        alpha: float = 0.05,
        mem_rank: int = 16,
        mem_scale: float = 4.0,
        dec_hidden: int = 64,
        use_prior: bool = True,
        vis_dim: int = 64,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d

        self.embed = nn.Embedding(vocab_size, emb_dim)

        # 시각 인코더
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, vis_dim)

        # PRISM 셀 (시각 항 포함)
        self.cell = PRISMMultimodalCell(
            d=d, emb_dim=emb_dim, K=K, alpha=alpha,
            memory_mode="sliding", mem_rank=mem_rank, mem_scale=mem_scale,
            decoder="mlp", dec_hidden=dec_hidden,
            use_prior=use_prior,
            vis_dim=vis_dim,
        )

        # u_rec: 텍스트 임베딩 + 이전 상태 융합
        self.u_rec1 = nn.Linear(d + emb_dim, emb_dim)
        self.u_rec2 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.output_proj = nn.Linear(d, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,                    # [B, T]
        images: Optional[torch.Tensor] = None,   # [B, T, C, H, W] or None
    ) -> Dict[str, Any]:
        B, T = tokens.shape
        device = tokens.device

        x, mem = self.cell.init_state(B, device)
        u_all = self.embed(tokens[:, :-1])        # [B, T-1, emb_dim]

        # 이미지 특징 사전 추출
        if images is not None:
            # images: [B, T-1, C, H, W] → [B, T-1, vis_dim]
            imgs = images[:, :-1]
            Bi, Ti, C, H, W = imgs.shape
            v_all = self.patch_embed(imgs.reshape(Bi * Ti, C, H, W))
            v_all = v_all.reshape(Bi, Ti, -1)
        else:
            v_all = None

        all_x = []
        for t in range(T - 1):
            u_raw = u_all[:, t]
            u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x], dim=-1))))

            v_t = v_all[:, t] if v_all is not None else None
            x_prior = self.cell.prior_mu(x) if self.cell.use_prior else None

            x, mem = self.cell(u, mem, x, training=self.training,
                               x_prior=x_prior, v_vis=v_t)
            rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            x = x * rms
            all_x.append(x)

        x_stack = torch.stack(all_x, dim=1)      # [B, T-1, d]
        logits  = self.output_proj(x_stack)       # [B, T-1, vocab_size]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        return {"loss": loss, "logits": logits}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
