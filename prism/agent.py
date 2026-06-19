"""
PRISM 에이전트 — 여러 모달리티 + 행동을 하나의 에너지 E로 통합.

설계철학:
  "지각·기억·추론·행동은 전부 같은 에너지 함수 E를 경사하강하는 것."

에너지 (텍스트 + 임의 개수 슬롯):

  E(x) = ½‖ũ_text − g(x)‖²_Π1        ← 텍스트 지각 (cell 내장)
       + Σ_m ½‖o_m − g_m(x)‖²_Π_m    ← 슬롯 m 지각/행동 (시각, 음향, 행동, …)
       + ½‖(I−M)x‖²_Π2               ← 기억
       + ½‖x − μ(x_prev)‖²_Π3        ← prior
       + ½λ‖x‖²                       ← 정규화

각 모달리티 = 에너지 항 1개. fusion layer 불필요 — K-step 하강이
모든 제약을 동시에 만족시키는 상태 x*를 찾는다.

슬롯의 두 가지 용법:
  - 지각 슬롯 (관측 o_m 제공): 추론 시에도 항이 살아있어 x를 끌어당김.
  - 행동 슬롯 (출력): 학습 시 목표 행동을 항으로 주입(에너지에 결합),
    추론 시 항을 끄고 g_act(x*)로 행동을 읽어냄.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any

from .cell import PRISMCell


class ModalSlot(nn.Module):
    """단일 모달리티 슬롯: 디코더 g_m: ℝ^d → ℝ^dim + precision Π_m."""

    def __init__(self, d: int, dim: int, kind: str = "linear", hidden: int = 64):
        super().__init__()
        self.kind = kind
        self.dim = dim
        if kind == "linear":
            self.W = nn.Parameter(torch.empty(dim, d))
            nn.init.normal_(self.W, std=0.02)
        else:  # mlp (비선형 디코더 → 비볼록 에너지)
            self.W1 = nn.Parameter(torch.empty(hidden, d))
            self.b1 = nn.Parameter(torch.zeros(hidden))
            self.W2 = nn.Parameter(torch.empty(dim, hidden))
            self.b2 = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.W1, std=0.02)
            nn.init.normal_(self.W2, std=0.02)
        self.log_pi = nn.Parameter(torch.zeros(dim))

    @property
    def pi(self) -> torch.Tensor:
        return F.softplus(self.log_pi)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """g_m(x): 상태 → 예측된 모달리티 특징."""
        if self.kind == "linear":
            return x @ self.W.t()
        h = torch.tanh(x @ self.W1.t() + self.b1)
        return h @ self.W2.t() + self.b2

    def dec_grad(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """J_{g_m}ᵀ · w  (에너지 그래디언트의 슬롯 기여분)."""
        if self.kind == "linear":
            return w @ self.W
        h = torch.tanh(x @ self.W1.t() + self.b1)
        b = w @ self.W2
        return (b * (1.0 - h * h)) @ self.W1


class PRISMAgentCell(PRISMCell):
    """
    PRISMCell + 임의 개수 모달리티 슬롯.

    텍스트(u) 항은 base cell이 담당하고, 추가 슬롯(시각·행동·…)은
    _neg_grad_E에 그래디언트 기여를 더한다. base의 검증된 텍스트+기억+prior
    경로를 그대로 재사용한다.
    """

    def __init__(self, slot_specs: List[dict], **kwargs):
        super().__init__(**kwargs)
        self.slots = nn.ModuleList([ModalSlot(self.d, **spec) for spec in slot_specs])

    def _neg_grad_E(self, x, u, mem_state, x_prior=None, obs=None, **kw):
        g = super()._neg_grad_E(x, u, mem_state, x_prior=x_prior)
        if obs is not None:
            for slot, o in zip(self.slots, obs):
                if o is not None:
                    eps = o - slot.decode(x)
                    g = g + slot.dec_grad(x, slot.pi * eps)
        return g

    def iterate(self, u, mem_state, x0=None, K=None, training=False,
                x_prior=None, obs=None, **kw):
        K = K if K is not None else self.K
        x = self.x_init(u) if x0 is None else x0
        with torch.set_grad_enabled(training):
            for _ in range(K):
                x = x + self.alpha * self._neg_grad_E(
                    self._rms(x), u, mem_state, x_prior=x_prior, obs=obs)
        return self._rms(x)

    def forward(self, u, mem_state, x0=None, training=True,
                x_prior=None, K=None, obs=None):
        x_star = self.iterate(u, mem_state, x0, K=K, training=training,
                              x_prior=x_prior, obs=obs)
        mem_new = self.update_memory(x_star.detach(), mem_state)
        return x_star, mem_new


class PatchEncoder(nn.Module):
    """이미지 → 특징 벡터 (시각 관측 o_vis).

    패치 그리드의 공간 구조를 보존해야 숫자(공간 패턴)를 구분할 수 있다.
    spatial pooling은 sin/cos 패턴을 평균내 정보를 파괴하므로 사용하지 않고,
    패치 그리드를 flatten 후 선형 투영한다.
    """

    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 vis_dim=64, hidden=64):
        super().__init__()
        assert img_size % patch_size == 0
        grid = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, hidden, patch_size, stride=patch_size)
        self.head = nn.Linear(hidden * grid * grid, vis_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj(img))          # [B, hidden, grid, grid]
        return self.head(x.flatten(1))      # [B, vis_dim]


class PRISMAgentModel(nn.Module):
    """
    텍스트 + 시각(지각) + 행동(출력)을 하나의 에너지 E로 처리하는 에이전트.

    처리 흐름 (토큰 t마다):
      1. u_t   = u_rec(Embed(s_t), x_prev)          텍스트 관측
      2. v_t   = Vision(img_t)                       시각 관측 (옵션)
      3. a_t   = onehot(action_target)              행동 관측 (학습 시 마지막 토큰만)
      4. x_t*  = argmin_x E(x; u_t, v_t, a_t, mem)  K-step 하강 (동시 만족)
      5. mem_t = Hebbian(x_t*)
      6. 읽기: text_logits = Unembed(x_t*),  action_logits = g_act(x_t*)

    슬롯 순서: slots[0]=vision, slots[1]=action.
    """

    SLOT_VISION = 0
    SLOT_ACTION = 1

    def __init__(
        self,
        vocab_size: int,
        n_actions: int,
        d: int = 128,
        emb_dim: int = 32,
        K: int = 4,
        alpha: float = 0.05,
        mem_rank: int = 8,
        mem_scale: float = 4.0,
        dec_hidden: int = 64,
        vis_dim: int = 32,
        img_size: int = 28,
        patch_size: int = 7,
        in_channels: int = 1,
        use_prior: bool = True,
        act_kind: str = "mlp",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.d = d

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.vision = PatchEncoder(img_size, patch_size, in_channels, vis_dim)

        self.cell = PRISMAgentCell(
            slot_specs=[
                dict(dim=vis_dim, kind="mlp", hidden=dec_hidden),   # 시각
                dict(dim=n_actions, kind=act_kind, hidden=dec_hidden),  # 행동
            ],
            d=d, emb_dim=emb_dim, K=K, alpha=alpha,
            memory_mode="sliding", mem_rank=mem_rank, mem_scale=mem_scale,
            decoder="mlp", dec_hidden=dec_hidden, use_prior=use_prior,
        )

        self.u_rec1 = nn.Linear(d + emb_dim, emb_dim)
        self.u_rec2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.output_proj = nn.Linear(d, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,                 # [B, T]
        images: Optional[torch.Tensor] = None,  # [B, C, H, W] (시퀀스 공통)
        action_labels: Optional[torch.Tensor] = None,  # [B]
        use_vision: bool = True,
        couple_action: bool = True,
    ) -> Dict[str, Any]:
        """
        use_vision   : 시각 에너지 항 on/off (ablation)
        couple_action: 학습 시 행동을 에너지에 결합할지 (False면 순수 readout head)
        """
        B, T = tokens.shape
        device = tokens.device

        x, mem = self.cell.init_state(B, device)
        u_all = self.embed(tokens[:, :-1] if T > 1 else tokens)

        v = self.vision(images) if (images is not None and use_vision) else None
        act_obs = None
        if (couple_action and self.training and action_labels is not None):
            act_obs = F.one_hot(action_labels, self.n_actions).float()

        steps = u_all.shape[1]
        all_x = []
        for t in range(steps):
            u_raw = u_all[:, t]
            u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x], dim=-1))))
            x_prior = self.cell.prior_mu(x) if self.cell.use_prior else None

            # 행동 항은 마지막 스텝에서만 결합 (행동은 시퀀스 단위 결정).
            a_t = act_obs if (t == steps - 1) else None
            obs = [v, a_t]

            x, mem = self.cell(u, mem, x, training=self.training,
                               x_prior=x_prior, obs=obs)
            x = x * x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            all_x.append(x)

        x_stack = torch.stack(all_x, dim=1)             # [B, steps, d]
        x_last = all_x[-1]                               # [B, d]

        result: Dict[str, Any] = {}

        # 텍스트 예측 (next-token)
        if T > 1:
            text_logits = self.output_proj(x_stack)
            text_loss = F.cross_entropy(
                text_logits.reshape(-1, self.vocab_size),
                tokens[:, 1:].reshape(-1))
            result["text_logits"] = text_logits
            result["text_loss"] = text_loss

        # 행동 읽기: g_act(x*) — 추론 시 행동 항은 꺼져 있음(act_obs=None at eval)
        action_logits = self.cell.slots[self.SLOT_ACTION].decode(x_last)
        result["action_logits"] = action_logits
        if action_labels is not None:
            result["action_loss"] = F.cross_entropy(action_logits, action_labels)
            result["action_acc"] = (action_logits.argmax(-1) == action_labels
                                    ).float().mean().item()
        return result

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
