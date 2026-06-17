"""
Mamba (S6) 베이스라인 — 순수 PyTorch, CPU 호환.

Mamba 핵심 아이디어:
  - 선택적 상태공간 모델 (Selective SSM / S6)
  - 입력에 따라 A, B, C 행렬이 달라짐 (selective)
  - h_t = A(x_t) h_{t-1} + B(x_t) x_t
  - y_t = C(x_t) h_t

CUDA 커널(parallel scan) 없이 순차 루프로 구현 → CPU에서 정확하게 동작.
파라미터 예산: ~55K (PRISM과 동일)

참고: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MambaBlock(nn.Module):
    """
    단일 Mamba 블록.

    x_in [B, d_model] → y [B, d_model]

    입력마다 SSM 파라미터(B, C, Δ)를 생성 → 선택적 상태 전이.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # 입력 투영
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 로컬 컨볼루션 (위치 인코딩 대체)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM 파라미터 생성기 (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, Δ
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # 고정 A (학습: log 스케일)
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 출력 투영
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 정규화
        self.norm = nn.LayerNorm(d_model)

    def ssm_step(self, x: torch.Tensor, h: torch.Tensor) -> tuple:
        """
        단일 토큰 SSM 스텝 (순차 모드).
        x: [B, d_inner]
        h: [B, d_inner, d_state]
        → y: [B, d_inner], h_new: [B, d_inner, d_state]
        """
        B_batch = x.shape[0]

        # 선택적 파라미터 생성
        xz = self.x_proj(x)                          # [B, 2*d_state + 1]
        B_ssm = xz[:, :self.d_state]                 # [B, d_state]
        C_ssm = xz[:, self.d_state:2*self.d_state]   # [B, d_state]
        dt = F.softplus(self.dt_proj(xz[:, -1:]))    # [B, d_inner]

        # A 이산화 (ZOH: A_bar = exp(Δ A))
        A = -torch.exp(self.A_log)                    # [d_inner, d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]

        # B 이산화 (Δ B x)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1)   # [B, d_inner, d_state]

        # 상태 전이: h = dA * h + dB * x
        h_new = dA * h + dB * x.unsqueeze(-1)        # [B, d_inner, d_state]

        # 출력: y = C h + D x
        y = (h_new * C_ssm.unsqueeze(1)).sum(-1) + self.D * x  # [B, d_inner]

        return y, h_new

    def forward(self, x_seq: torch.Tensor, h0=None) -> tuple:
        """
        x_seq: [B, T, d_model]
        h0: [B, d_inner, d_state] or None
        → out: [B, T, d_model], h_final
        """
        B, T, _ = x_seq.shape
        residual = x_seq

        x_seq = self.norm(x_seq)

        # 입력 분할: x와 z (gating)
        xz = self.in_proj(x_seq)                     # [B, T, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)                   # [B, T, d_inner] each

        # 로컬 컨볼루션 (T 차원)
        x = x.transpose(1, 2)                        # [B, d_inner, T]
        x = self.conv1d(x)[:, :, :T]                 # causal: 앞 T만
        x = x.transpose(1, 2)                        # [B, T, d_inner]
        x = F.silu(x)

        # 순차 SSM
        if h0 is None:
            h = torch.zeros(B, self.d_inner, self.d_state, device=x_seq.device)
        else:
            h = h0

        outs = []
        for t in range(T):
            y_t, h = self.ssm_step(x[:, t], h)
            outs.append(y_t)

        y = torch.stack(outs, dim=1)                 # [B, T, d_inner]
        y = y * F.silu(z)                            # gating

        out = self.out_proj(y)                       # [B, T, d_model]
        return out + residual, h


class MambaLangModel(nn.Module):
    """
    Mamba 언어 모델 (PRISM 파라미터 예산 매칭용).

    단일 Mamba 블록 + embed + unembed.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.constant_(self.mamba.dt_proj.bias, math.log(math.expm1(1.0)))

    def forward(self, tokens: torch.Tensor, h0=None):
        """tokens: [B, T] → {'loss', 'logits', 'h_final'}"""
        x = self.embed(tokens[:, :-1])               # [B, T-1, d_model]
        y, h = self.mamba(x, h0)                     # [B, T-1, d_model]
        y = self.norm_f(y)
        logits = self.output_proj(y)                 # [B, T-1, vocab_size]

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens[:, 1:].reshape(-1),
        )
        return {'loss': loss, 'logits': logits, 'h_final': h}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
