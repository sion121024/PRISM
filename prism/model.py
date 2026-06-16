"""
PRISM language model: character-level autoregressive LM built on PRISMCell.

Architecture:
  embed  : token  -> u ∈ ℝ^d
  cell   : (x, u) -> x*  (energy descent, K inner steps)
  head   : x*     -> logits over vocab

DEQ implicit differentiation for training (memory O(1) vs BPTT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .energy import PRISMCell


# ------------------------------------------------------------------ #
#  DEQ implicit diff backward hook                                    #
# ------------------------------------------------------------------ #

class _DEQBackward(torch.autograd.Function):
    """
    Given fixed-point z* s.t. f(z*; θ) = z*, compute ∂L/∂θ via:
      ∂L/∂θ = (∂L/∂z*) · (I - ∂f/∂z*)⁻¹ · ∂f/∂θ
    We approximate (I - J)⁻¹ with a fixed number of Neumann iterations.
    """

    @staticmethod
    def forward(ctx, z_star, f_thunk, neumann_iters=5):
        ctx.save_for_backward(z_star)
        ctx.f_thunk = f_thunk
        ctx.neumann_iters = neumann_iters
        return z_star

    @staticmethod
    def backward(ctx, grad_output):
        z_star, = ctx.saved_tensors
        f_thunk = ctx.f_thunk
        n = ctx.neumann_iters

        # Neumann series: (I - J)⁻¹ v ≈ v + Jv + J²v + ...
        v = grad_output.clone()
        neumann = v.clone()
        with torch.enable_grad():
            z = z_star.detach().requires_grad_(True)
            fz = f_thunk(z)
        for _ in range(n):
            v = torch.autograd.grad(
                fz, z, grad_outputs=v, retain_graph=True
            )[0]
            neumann = neumann + v
        return neumann, None, None


def deq_backward(z_star, f_thunk, neumann_iters=5):
    return _DEQBackward.apply(z_star, f_thunk, neumann_iters)


# ------------------------------------------------------------------ #
#  Full PRISM LM                                                      #
# ------------------------------------------------------------------ #

class PRISMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d: int = 256,
        K: int = 8,
        step: float = 0.1,
        lam: float = 0.01,
        neumann_iters: int = 5,
    ):
        super().__init__()
        self.d = d
        self.K = K
        self.step = step
        self.neumann_iters = neumann_iters

        self.embed = nn.Embedding(vocab_size, d)
        self.cell  = PRISMCell(d, lam=lam)
        self.head  = nn.Linear(d, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight,  std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, T)  int64
        K: int | None = None,
        use_deq: bool = True,
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size)."""
        K = K or self.K
        B, T = tokens.shape
        device = tokens.device

        x = torch.zeros(B, self.d, device=device)
        self.cell.reset_memory()

        logits_list = []
        for t in range(T):
            u = self.embed(tokens[:, t])  # (B, d)

            if use_deq and self.training:
                # DEQ: run forward to fixed point, then implicit diff
                with torch.no_grad():
                    x_star, _ = self.cell(x.detach(), u, K=K, step=self.step)

                # re-run ONE step with grad enabled to get Jacobian hook
                def f_thunk(z):
                    g = self.cell.grad_E(z, u)
                    return z - self.step * g

                x_star = deq_backward(
                    x_star.detach().requires_grad_(True),
                    f_thunk,
                    self.neumann_iters,
                )
                x = x_star
            else:
                x, _ = self.cell(x, u, K=K, step=self.step)

            logits_list.append(self.head(x))   # (B, vocab)

        return torch.stack(logits_list, dim=1)  # (B, T, vocab)

    def generate(
        self,
        prompt: torch.Tensor,    # (1, T_prompt)
        max_new: int = 200,
        temperature: float = 1.0,
        K: int | None = None,
        top_k: int = 0,
    ) -> list[int]:
        self.eval()
        K = K or self.K
        device = prompt.device
        x = torch.zeros(1, self.d, device=device)
        self.cell.reset_memory()

        # Warm up on prompt
        with torch.no_grad():
            for t in range(prompt.shape[1]):
                u = self.embed(prompt[:, t])
                x, _ = self.cell(x, u, K=K, step=self.step)

            generated = []
            tok = prompt[:, -1]
            for _ in range(max_new):
                u = self.embed(tok)
                x, _ = self.cell(x, u, K=K, step=self.step)
                logits = self.head(x) / temperature
                if top_k > 0:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, -1:]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                tok = torch.multinomial(probs, 1)
                generated.append(tok.item())

        return generated
