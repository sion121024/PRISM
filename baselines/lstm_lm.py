"""
LSTM 베이스라인 언어 모델 (PRISM 비교용).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class LSTMLangModel(nn.Module):
    """단순 LSTM 기반 언어 모델."""

    def __init__(
        self,
        vocab_size: int = 256,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,       # [B, T]
        hidden: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        B, T = tokens.shape
        emb = self.embed(tokens[:, :-1])          # [B, T-1, emb_dim]
        out, hidden = self.lstm(emb, hidden)       # [B, T-1, hidden]
        logits = self.output_proj(out)             # [B, T-1, vocab]
        targets = tokens[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        return {'loss': loss, 'logits': logits, 'hidden': hidden}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
