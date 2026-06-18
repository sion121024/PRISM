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
from typing import Optional, List, Dict, Any

from .cell import PRISMCell
from .deq import DEQSolver


class PRISMLangModel(nn.Module):
    """
    PRISM 언어 모델.

    처리 흐름 (토큰 t마다):
      1. u_t = Embed(s_t)                              [optional: conv + urec]
      2. x_t* = iterate(u_t, mem_{t-1}, x_{t-1})      [내부 K회 에너지 하강]
      3. mem_t = update_memory(x_t*)                   [Hebbian 빠른가중치]
      4. h = norm_f(x_t*) [* gate(u)]                 [readout, gate는 상태 수정 없음]
      5. logits = output_proj(h)
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
        input_dep_pi: bool = False,
        momentum: float = 0.0,
        use_conv: bool = False,
        d_conv: int = 4,
        use_gate: bool = False,  # readout-only gate: h = norm(x) * SiLU(W_g u)
        use_compile: bool = False,
        n_layers: int = 1,       # 계층적 예측 코딩 레이어 수 (>1 → 다층 PRISM)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.emb_dim = emb_dim
        self.use_deq = use_deq
        self.use_prior = use_prior
        self.simple_prior = simple_prior
        self.use_urec = use_urec
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.use_gate = use_gate
        self.n_layers = n_layers

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
            input_dep_pi=input_dep_pi,
            momentum=momentum,
        )
        self.norm_f = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, vocab_size, bias=False)

        # 계층적 예측 코딩 상위 레이어들.
        # 레이어 l>0은 레이어 l-1의 상태(d-dim)를 관측값으로 받아 자신의 에너지를 하강.
        if n_layers > 1:
            self.upper_cells = nn.ModuleList([
                PRISMCell(
                    d=d, emb_dim=d,
                    lam=lam, K=K, alpha=alpha,
                    mem_eta=mem_eta, mem_gamma=mem_gamma,
                    memory_mode=memory_mode, mem_rank=mem_rank,
                    approximate_grad=approximate_grad,
                    mem_scale=mem_scale, state_norm=state_norm,
                    simple_prior=True,   # identity prior: μ = x_{l,prev}
                    momentum=momentum,
                )
                for _ in range(n_layers - 1)
            ])
        else:
            self.upper_cells = nn.ModuleList([])

        if use_conv:
            self.conv_1d = nn.Conv1d(
                emb_dim, emb_dim, kernel_size=d_conv,
                padding=d_conv - 1, groups=emb_dim, bias=True,
            )

        if use_gate:
            # 출력 게이트: norm(x_t*)에만 적용, 재귀 상태 x_t*에는 적용하지 않음.
            # bias=1.278 → SiLU(1.278) ≈ 1.0, 초기에는 identity-like.
            self.gate_proj = nn.Linear(emb_dim, d)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 1.278)

        if use_urec:
            self.u_rec1 = nn.Linear(d + emb_dim, emb_dim)
            self.u_rec2 = nn.Linear(emb_dim, emb_dim, bias=False)

        self.carry_nonlin = carry_nonlin
        if carry_nonlin:
            self.carry_gate = nn.Sequential(
                nn.Linear(d, d // 4),
                nn.GELU(),
                nn.Linear(d // 4, d),
            )
            self.carry_ln = nn.LayerNorm(d)

        if use_deq:
            self.deq = DEQSolver()

        if use_compile and hasattr(torch, 'compile'):
            try:
                compiled = torch.compile(
                    self.cell.iterate, mode="reduce-overhead", dynamic=True)
                import types  # noqa: F401
                self.cell.iterate = compiled
            except Exception:
                pass

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()

    def init_state(self, batch_size: int, device: torch.device):
        x0, mem0 = self.cell.init_state(batch_size, device)
        if self.n_layers == 1:
            return x0, mem0
        # 다층: all_xs[0..n-1], all_mems[0..n-1].  외부에는 (x_top, packed_mem) 노출.
        all_xs = [x0] + [
            torch.zeros(batch_size, self.d, device=device)
            for _ in range(self.n_layers - 1)
        ]
        all_mems = [mem0] + [
            self.upper_cells[l].init_state(batch_size, device)[1]
            for l in range(self.n_layers - 1)
        ]
        return all_xs[-1], (all_xs, all_mems)

    # ---------------------------------------------------------------- #
    # Forward                                                           #
    # ---------------------------------------------------------------- #

    def _token_K(self, prev_logits: torch.Tensor, K_min: int, K_max: int) -> int:
        """이전 예측 엔트로피 → 현재 토큰 K 결정 (이중시계 적응)."""
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
        prev_logits: Optional[torch.Tensor] = None

        # 사전 임베딩: raw (게이트/urec 기준) + processed (conv 적용 후 cell 관측값)
        u_all_raw = self.embed(tokens[:, :-1])   # [B, T-1, emb_dim]
        u_all = u_all_raw
        if self.use_conv:
            u_t = u_all_raw.transpose(1, 2)
            u_t = self.conv_1d(u_t)[:, :, :T - 1]
            u_all = F.silu(u_t.transpose(1, 2))  # [B, T-1, emb_dim]

        # ---- 다층 분기 ---- #
        if self.n_layers > 1:
            if isinstance(mem, tuple) and isinstance(mem[0], list):
                xs, mems = mem  # 외부에서 전달된 패킹 상태
            else:
                _, (xs, mems) = self.init_state(B, device)

            for t in range(T - 1):
                u_raw_t = u_all[:, t]
                u_embed_t = u_all_raw[:, t]

                # Layer 0: token embedding → state
                if self.use_urec:
                    u0 = self.u_rec2(F.gelu(self.u_rec1(
                        torch.cat([u_raw_t, xs[0]], dim=-1))))
                else:
                    u0 = u_raw_t
                x_prior0 = (xs[0] if self.simple_prior else
                            (self.cell.prior_mu(xs[0]) if self.use_prior else None))
                xs[0], mems[0] = self.cell(u0, mems[0], xs[0],
                                           training=is_training, x_prior=x_prior0)
                xs[0] = self._rms_norm(xs[0])

                # Layers 1+: lower state → upper state (계층적 예측 코딩)
                for l in range(self.n_layers - 1):
                    x_obs = xs[l]           # 하위 레이어 출력 = 관측값
                    x_prior_l = xs[l + 1]   # 상위 레이어 이전 상태 = identity prior
                    xs[l + 1], mems[l + 1] = self.upper_cells[l](
                        x_obs, mems[l + 1], xs[l + 1],
                        training=is_training, x_prior=x_prior_l)
                    xs[l + 1] = self._rms_norm(xs[l + 1])

                all_x.append(xs[-1])

                if adaptive_K and not is_training:
                    prev_logits = self.output_proj(self.norm_f(xs[-1].detach()))
                    all_k_used.append(self.cell.K)

                if tbptt_window > 0 and (t + 1) % tbptt_window == 0:
                    xs = [xi.detach() for xi in xs]

        # ---- 단층 분기 ---- #
        else:
            for t in range(T - 1):
                u_raw_t = u_all[:, t]
                u_embed_t = u_all_raw[:, t]

                if self.use_urec:
                    u = self.u_rec2(F.gelu(self.u_rec1(
                        torch.cat([u_raw_t, x], dim=-1))))
                else:
                    u = u_raw_t

                if adaptive_K and not is_training and prev_logits is not None:
                    k_t = self._token_K(prev_logits, K_min, _K_max)
                else:
                    k_t = None

                if return_energies:
                    x0_cell = None if t == 0 else x
                    x_prior_e = (x if self.simple_prior else
                                 (self.cell.prior_mu(x) if self.use_prior else None))
                    x_new, energies_t = self.cell.iterate(
                        u, mem, x0_cell, return_energies=True,
                        training=False, x_prior=x_prior_e)
                    mem = self.cell.update_memory(x_new.detach(), mem)
                    x = self._rms_norm(x_new)
                    all_energies.append(energies_t)
                elif self.use_deq:
                    x_prior_d = (x if self.simple_prior else
                                 (self.cell.prior_mu(x) if self.use_prior else None))
                    F_fn = lambda z: z + self.cell.alpha * self.cell._neg_grad_E(
                        z, u, mem, x_prior=x_prior_d)
                    x_star, _ = self.deq(F_fn, x, list(self.cell.parameters()))
                    mem = self.cell.update_memory(x_star.detach(), mem)
                    x = self._rms_norm(x_star)
                else:
                    x_prior = (x if self.simple_prior else
                               (self.cell.prior_mu(x) if self.use_prior else None))
                    x0_cell = None if t == 0 else x
                    x, mem = self.cell(u, mem, x0_cell, training=is_training,
                                       x_prior=x_prior, K=k_t)
                    if self.carry_nonlin:
                        x = self.carry_ln(x + self.carry_gate(x))
                    else:
                        x = self._rms_norm(x)

                if adaptive_K and not is_training:
                    prev_logits = self.output_proj(self.norm_f(x.detach()))
                    all_k_used.append(k_t if k_t is not None else self.cell.K)

                all_x.append(x)

                if tbptt_window > 0 and (t + 1) % tbptt_window == 0:
                    x = x.detach()

        # ---- 배치 출력 투영 ---- #
        x_stacked = torch.stack(all_x, dim=1)           # [B, T-1, d]
        h = self.norm_f(x_stacked)                      # [B, T-1, d]
        if self.use_gate:
            # readout gate: h에만 적용. 재귀 상태 x_t*는 변경하지 않음.
            h = h * F.silu(self.gate_proj(u_all_raw))   # [B, T-1, d]
        logits_all = self.output_proj(h)                # [B, T-1, vocab_size]

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

        # 다층: 내부 상태 언패킹
        if self.n_layers > 1:
            _, (xs, mems) = self.init_state(B, device)

        conv_buf = x.new_zeros(B, self.emb_dim, self.d_conv) if self.use_conv else None

        def _apply_conv(u_embed, buf):
            buf = torch.cat([buf[:, :, 1:], u_embed.unsqueeze(-1)], dim=-1)
            u_c = (self.conv_1d.weight.squeeze(1) * buf).sum(-1) + self.conv_1d.bias
            return F.silu(u_c), buf

        def _cell_step(u_embed, x_in, mem_in, x0_arg):
            """단층 1-step. u_embed: pre-conv 임베딩."""
            nonlocal conv_buf
            if self.use_conv:
                u_raw, conv_buf = _apply_conv(u_embed, conv_buf)
            else:
                u_raw = u_embed
            if self.use_urec:
                x_ref = x_in if x_in is not None else torch.zeros(B, self.d, device=device)
                u = self.u_rec2(F.gelu(self.u_rec1(torch.cat([u_raw, x_ref], dim=-1))))
            else:
                u = u_raw
            x_prior = (x_in if self.simple_prior else
                       (self.cell.prior_mu(x_in) if self.use_prior and x_in is not None else None))
            x_new = self.cell.iterate(u, mem_in, x0_arg, K=K_gen, training=False, x_prior=x_prior)
            mem_new = self.cell.update_memory(x_new, mem_in)
            x_new = self._rms_norm(x_new)
            return x_new, mem_new, u_embed   # u_embed 반환 (게이트용)

        def _upper_step(x_prev_in, xs_in, mems_in):
            """다층 상위 레이어 1-step."""
            for l in range(self.n_layers - 1):
                xs_in[l + 1], mems_in[l + 1] = self.upper_cells[l](
                    xs_in[l], mems_in[l + 1], xs_in[l + 1],
                    training=False, x_prior=xs_in[l + 1])
                xs_in[l + 1] = self._rms_norm(xs_in[l + 1])

        # 프롬프트 처리
        _first = True
        last_u_embed = None
        for tok in prompt.unbind(1):
            u_embed = self.embed(tok)
            if self.n_layers > 1:
                if self.use_conv:
                    u_raw, conv_buf = _apply_conv(u_embed, conv_buf)
                else:
                    u_raw = u_embed
                if self.use_urec:
                    u = self.u_rec2(F.gelu(self.u_rec1(
                        torch.cat([u_raw, xs[0]], dim=-1))))
                else:
                    u = u_raw
                x_prior0 = (xs[0] if self.simple_prior else
                            (self.cell.prior_mu(xs[0]) if self.use_prior else None))
                xs[0] = self.cell.iterate(u, mems[0], xs[0] if not _first else None,
                                          K=K_gen, training=False, x_prior=x_prior0)
                mems[0] = self.cell.update_memory(xs[0], mems[0])
                xs[0] = self._rms_norm(xs[0])
                _upper_step(xs[0], xs, mems)
            else:
                x, mem, u_embed = _cell_step(u_embed, x if not _first else None, mem, None if _first else x)
            _first = False
            last_u_embed = u_embed

        # 생성 루프
        generated = prompt.tolist()
        for _ in range(max_new_tokens):
            if self.n_layers > 1:
                h = self.norm_f(xs[-1])
                if self.use_gate and last_u_embed is not None:
                    h = h * F.silu(self.gate_proj(last_u_embed))
                logits = self.output_proj(h)
            else:
                h = self.norm_f(x)
                if self.use_gate and last_u_embed is not None:
                    h = h * F.silu(self.gate_proj(last_u_embed))
                logits = self.output_proj(h)

            logits = logits / temperature
            if top_k is not None:
                topk_val = torch.topk(logits, top_k, dim=-1).values
                logits = logits.masked_fill(logits < topk_val[:, -1:], -1e9)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            for b in range(B):
                generated[b].append(next_tok[b].item())

            u_embed = self.embed(next_tok)
            last_u_embed = u_embed
            if self.n_layers > 1:
                if self.use_conv:
                    u_raw, conv_buf = _apply_conv(u_embed, conv_buf)
                else:
                    u_raw = u_embed
                if self.use_urec:
                    u = self.u_rec2(F.gelu(self.u_rec1(
                        torch.cat([u_raw, xs[0]], dim=-1))))
                else:
                    u = u_raw
                x_prior0 = (xs[0] if self.simple_prior else
                            (self.cell.prior_mu(xs[0]) if self.use_prior else None))
                xs[0] = self.cell.iterate(u, mems[0], xs[0], K=K_gen, training=False, x_prior=x_prior0)
                mems[0] = self.cell.update_memory(xs[0], mems[0])
                xs[0] = self._rms_norm(xs[0])
                _upper_step(xs[0], xs, mems)
            else:
                x, mem, _ = _cell_step(u_embed, x, mem, x)

        return torch.tensor(generated, device=device)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
