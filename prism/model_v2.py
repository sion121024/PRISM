"""PRISM-Base LM (v2): PRISMBaseCell with Nesterov + low-rank M."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .energy_v2 import PRISMBaseCell


class PRISMBase(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d: int = 256,
        rank: int | None = None,
        K: int = 6,
        step: float = 0.1,
        momentum: float = 0.9,
        lam: float = 0.01,
    ):
        super().__init__()
        self.d        = d
        self.K        = K
        self.step     = step
        self.momentum = momentum

        self.embed = nn.Embedding(vocab_size, d)
        self.cell  = PRISMBaseCell(d, rank=rank, lam=lam)
        self.head  = nn.Linear(d, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight,  std=0.02)

    def forward(self, tokens: torch.Tensor, K: int | None = None) -> torch.Tensor:
        K = K or self.K
        B, T = tokens.shape
        x = torch.zeros(B, self.d, device=tokens.device)
        self.cell.reset_memory()

        logits_list = []
        for t in range(T):
            u = self.embed(tokens[:, t])
            x, _ = self.cell(x, u, K=K, step=self.step, momentum=self.momentum)
            logits_list.append(self.head(x))

        return torch.stack(logits_list, dim=1)

    def generate(
        self, prompt, max_new=200, temperature=1.0, K=None, top_k=40
    ) -> list[int]:
        self.eval()
        K = K or self.K
        x = torch.zeros(1, self.d, device=prompt.device)
        self.cell.reset_memory()

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
                tok = torch.multinomial(F.softmax(logits, -1), 1)
                generated.append(tok.item())

        return generated
