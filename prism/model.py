"""
PRISM 언어 모델.

외부시계 t: 토큰 순서 처리
내부시계 s: 각 토큰에서 K번 에너지 하강

memory_mode: 'sliding' (CPU 기본) | 'full_M' (GPU 권장) | 'none'
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from .cell import PRISMCell
from .deq import DEQSolver


class PRISMLangModel(nn.Module):
    """
    PRISM 언어 모델.

    처리 흐름 (토큰 t마다):
      1. u_t = Embed(s_t)
      2. x_t* = iterate(u_t, mem_{t-1}, x_{t-1})  [내부 K회]
      3. mem_t = update_memory(x_t*)                [Hebbian]
      4. logits = Unembed(x_t*)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d: int = 256,
        emb_dim: int = 64,
        K: int = 16,
        alpha: float = 0.05,
        lam: float = 0.01,
        mem_eta: float = 0.01,
        mem_gamma: float = 0.001,
        use_deq: bool = False,
        memory_mode: str = "sliding",
        mem_rank: int = 32,
        approximate_grad: bool = False,
        decoder: str = "linear",
        dec_hidden: int = 128,
        mem_scale: float = 1.0,
        carry_nonlin: bool = False,
        state_norm: Optional[bool] = None,
        use_prior: bool = False,
        use_urec: bool = True,
        simple_prior: bool = False,
        prior_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.emb_dim = emb_dim
        self.use_deq = use_deq
        self.use_prior = use_prior
        self.simple_prior = simple_prior
        self.use_urec = use_urec

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.cell = PRISMCell(
            d=d, emb_dim=emb_dim, lam=lam,
            K=K, alpha=alpha,
            mem_eta=mem_eta, mem_gamma=mem_gamma,
            memory_mode=memory_mode,
            mem_rank=mem_rank,
            approximate_grad=approximate_grad,
            decoder=decoder,
            dec_hidden=dec_hidden,
            mem_scale=mem_scale,
            state_norm=state_norm,
            use_prior=use_prior,
            simple_prior=simple_prior,
            prior_bias=prior_bias,
        )
        self.output_proj = nn.Linear(d, vocab_size, bias=False)

        if use_urec:
            # 설계 정합 순환: ũ = f([u, x_prev]) → 에너지 관측값 u를 풍부하게.
            # carry gate(에너지 밖 변환) 대신 사용.
            self.u_rec1 = nn.Linear(d + emb_dim, emb_dim)
            self.u_rec2 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.carry_nonlin = carry_nonlin
        if carry_nonlin:
            # carry gate는 하위호환용으로만 유지 (설계 비정합, 신규 코드에서 사용 금지)
            self.carry_gate = nn.Sequential(
                nn.Linear(d, d // 4),
                nn.GELU(),
                nn.Linear(d // 4, d),
            )
            self.carry_ln = nn.LayerNorm(d)

        if use_deq:
            self.deq = DEQSolver()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def init_state(self, batch_size: int, device: torch.device):
        return self.cell.init_state(batch_size, device)

    # ---------------------------------------------------------------- #
    # Forward                                                           #
    # ---------------------------------------------------------------- #

    def _token_K(self, prev_logits: torch.Tensor, K_min: int, K_max: int) -> int:
        """
        이전 토큰 예측 엔트로피 → 현재 토큰 K 결정.
        엔트로피 높음(불확실) = K 많이, 엔트로피 낮음(확실) = K 적게.
        설계 이중시계: 어려운 입력일수록 내부시계가 더 많이 돈다.
        """
        with torch.no_grad():
            probs = F.softmax(prev_logits, dim=-1)
            entropy = -(probs * (probs + 1e-9).log()).sum(-1).max().item()
            max_entropy = math.log(self.vocab_size)
            frac = min(entropy / max_entropy, 1.0)
            return max(K_min, min(K_max, K_min + round((K_max - K_min) * frac)))

    def forward(
        self,
        tokens: torch.Tensor,            # [B, T]
        x0=None,
        mem0=None,
        return_energies: bool = False,
        tbptt_window: int = 0,
        adaptive_K: bool = False,
        K_min: int = 1,
        K_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        B, T = tokens.shape
        device = tokens.device

        x, mem = self.init_state(B, device)
        if x0 is not None:
            x = x0
        if mem0 is not None:
            mem = mem0

        is_training = self.training
        all_x: List[torch.Tensor] = []
        all_energies: List[List[float]] = []
        all_k_used: List[int] = []

        _K_max = K_max if K_max is not None else self.cell.K * 2
        prev_logits: Optional[torch.Tensor] = None  # 적응형 K용 이전 예측

        # Pre-embed all input tokens in one batched call
        u_all = self.embed(tokens[:, :-1])  # [B, T-1, emb_dim]

        for t in range(T - 1):
            u_raw = u_all[:, t]
            if self.use_urec:
                u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x], dim=-1))))
            else:
                u = u_raw

            # 적응형 K: 이전 토큰 예측 엔트로피로 현재 K 결정
            if adaptive_K and not is_training and prev_logits is not None:
                k_t = self._token_K(prev_logits, K_min, _K_max)
            else:
                k_t = None  # cell 기본값 사용

            if return_energies:
                x_new, energies_t = self.cell.iterate(
                    u, mem, x, return_energies=True, training=False)
                mem = self.cell.update_memory(x_new.detach(), mem)
                x = x_new
                all_energies.append(energies_t)
            elif self.use_deq:
                params = list(self.cell.parameters())
                F_fn = lambda z: z + self.cell.alpha * self.cell.neg_grad_E(z, u, mem)
                x_star, _ = self.deq(F_fn, x, params)
                mem = self.cell.update_memory(x_star.detach(), mem)
                x = x_star
            else:
                if self.simple_prior:
                    x_prior = x  # identity prior: μ = x_prev (파라미터 없음)
                elif self.use_prior:
                    x_prior = self.cell.prior_mu(x)
                else:
                    x_prior = None
                x, mem = self.cell(u, mem, x, training=is_training,
                                   x_prior=x_prior, K=k_t)
                if self.carry_nonlin:
                    x = self.carry_ln(x + self.carry_gate(x))
                else:
                    rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
                    x = x * rms

            # 다음 토큰 K 결정을 위해 현재 logits 저장
            if adaptive_K and not is_training:
                prev_logits = self.output_proj(x.detach())
                all_k_used.append(k_t if k_t is not None else self.cell.K)

            all_x.append(x)

            if tbptt_window > 0 and (t + 1) % tbptt_window == 0:
                x = x.detach()

        # Batch output projection: one call instead of T-1 calls
        x_stacked = torch.stack(all_x, dim=1)          # [B, T-1, d]
        logits_all = self.output_proj(x_stacked)        # [B, T-1, vocab_size]

        targets = tokens[:, 1:]
        loss = F.cross_entropy(
            logits_all.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        result: Dict[str, Any] = {'loss': loss, 'logits': logits_all}
        if return_energies:
            result['energies'] = all_energies
        if all_k_used:
            result['k_used'] = all_k_used
        return result

    # ---------------------------------------------------------------- #
    # 생성                                                              #
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        K_gen: Optional[int] = None,
    ) -> torch.Tensor:
        B = prompt.shape[0]
        device = prompt.device
        x, mem = self.init_state(B, device)

        for tok in prompt.unbind(1):
            u_raw = self.embed(tok)
            if self.use_urec:
                u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x], dim=-1))))
            else:
                u = u_raw
            x_prior = x if self.simple_prior else (self.cell.prior_mu(x) if self.use_prior else None)
            x = self.cell.iterate(u, mem, x, K=K_gen, training=False, x_prior=x_prior)
            mem = self.cell.update_memory(x, mem)
            rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            x = x * rms

        generated = prompt.tolist()
        for _ in range(max_new_tokens):
            logits = self.output_proj(x) / temperature
            if top_k is not None:
                topk_val = torch.topk(logits, top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_val[:, -1:], -1e9)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            for b in range(B):
                generated[b].append(next_tok[b].item())
            u_raw = self.embed(next_tok)
            if self.use_urec:
                u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x], dim=-1))))
            else:
                u = u_raw
            x_prior = x if self.simple_prior else (self.cell.prior_mu(x) if self.use_prior else None)
            x = self.cell.iterate(u, mem, x, K=K_gen, training=False, x_prior=x_prior)
            mem = self.cell.update_memory(x, mem)
            rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            x = x * rms

        return torch.tensor(generated, device=device)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
