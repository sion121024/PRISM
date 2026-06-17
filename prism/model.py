"""
PRISM 언어 모델.

외부시계 t: 토큰 순서 처리
내부시계 s: 각 토큰에서 K번 에너지 하강

memory_mode: 'sliding' (CPU 기본) | 'full_M' (GPU 권장) | 'none'
"""

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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.emb_dim = emb_dim
        self.use_deq = use_deq

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
        )
        self.output_proj = nn.Linear(d, vocab_size, bias=False)

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

    def forward(
        self,
        tokens: torch.Tensor,            # [B, T]
        x0=None,
        mem0=None,
        return_energies: bool = False,
        tbptt_window: int = 0,
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

        # Pre-embed all input tokens in one batched call
        u_all = self.embed(tokens[:, :-1])  # [B, T-1, emb_dim]

        for t in range(T - 1):
            u = u_all[:, t]

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
                x, mem = self.cell(u, mem, x, training=is_training)
                if self.carry_nonlin:
                    x = self.carry_ln(x + self.carry_gate(x))

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
            u = self.embed(tok)
            x = self.cell.iterate(u, mem, x, K=K_gen, training=False)
            mem = self.cell.update_memory(x, mem)

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
            u = self.embed(next_tok)
            x = self.cell.iterate(u, mem, x, K=K_gen, training=False)
            mem = self.cell.update_memory(x, mem)

        return torch.tensor(generated, device=device)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
