"""Simple GRU baseline for comparison with PRISM."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULM(nn.Module):
    def __init__(self, vocab_size: int, d: int = 256, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.rnn   = nn.GRU(d, d, num_layers=n_layers, batch_first=True)
        self.head  = nn.Linear(d, vocab_size, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight,  std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)          # (B, T, d)
        h, _ = self.rnn(x)              # (B, T, d)
        return self.head(h)             # (B, T, vocab)

    def generate(self, prompt, max_new=200, temperature=1.0, top_k=40):
        self.eval()
        device = prompt.device
        h = None
        with torch.no_grad():
            emb = self.embed(prompt)
            _, h = self.rnn(emb, h)
            generated = []
            tok = prompt[:, -1:]
            for _ in range(max_new):
                emb = self.embed(tok)
                out, h = self.rnn(emb, h)
                logits = self.head(out[:, -1]) / temperature
                if top_k > 0:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, -1:]] = -float("inf")
                tok = torch.multinomial(F.softmax(logits, -1), 1)
                generated.append(tok.item())
        return generated
