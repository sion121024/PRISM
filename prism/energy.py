"""
PRISM core: energy function and state dynamics.

E(x) = ½‖u - D·x‖²_Π1  +  ½‖(I-M)·x‖²_Π2  +  ½λ‖x‖²
dx/ds = -∂E/∂x  (repeated K times = "thinking")
ΔM    = η·(ε_mem·xᵀ) - γ·M  (fast-weight / test-time memory)
"""

import torch
import torch.nn as nn


class PRISMCell(nn.Module):
    """
    Single PRISM cell operating on state x ∈ ℝ^d.

    Outer clock tick  t : one token / frame  (O(1) fixed-size state)
    Inner clock steps s : K(t) gradient descent steps on E  (adaptive thinking depth)
    """

    def __init__(self, d: int, lam: float = 0.01):
        super().__init__()
        self.d = d
        self.lam = lam

        # Slow weights θ (learned via DEQ implicit diff)
        self.D = nn.Linear(d, d, bias=False)   # decoder: x -> observation space
        self.log_pi1 = nn.Parameter(torch.zeros(d))  # log precision for perception
        self.log_pi2 = nn.Parameter(torch.zeros(d))  # log precision for memory

        # Fast weight M updated at test-time (starts at 0 each sequence or carried)
        # Stored as buffer, NOT a parameter — updated by Hebbian rule
        self.register_buffer("M", torch.zeros(d, d))

        # Adaptive halting: scalar network maps energy-delta → halt prob
        self.halt_net = nn.Linear(1, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.D.weight, gain=0.5)
        nn.init.constant_(self.halt_net.bias, -2.0)  # bias toward continuing

    # ------------------------------------------------------------------ #
    #  Energy and its gradient                                            #
    # ------------------------------------------------------------------ #

    def energy(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Scalar energy per sample. x: (B,d), u: (B,d)"""
        pi1 = self.log_pi1.exp()   # (d,)
        pi2 = self.log_pi2.exp()

        eps_in  = u - x @ self.D.weight.T          # (B,d)  perception error
        eps_mem = x - x @ self.M.T                  # (B,d)  memory error: (I-Mᵀ)x

        E = (0.5 * (eps_in**2 * pi1).sum(-1)
           + 0.5 * (eps_mem**2 * pi2).sum(-1)
           + 0.5 * self.lam * (x**2).sum(-1))
        return E  # (B,)

    def grad_E(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """∂E/∂x analytically (no autograd needed in forward pass)."""
        pi1 = self.log_pi1.exp()
        pi2 = self.log_pi2.exp()

        eps_in  = u - x @ self.D.weight.T
        eps_mem = x - x @ self.M.T

        # -DᵀΠ1·εin  +  (I-M)ᵀΠ2·εmem  +  λx
        g = (- (eps_in * pi1) @ self.D.weight
             + (eps_mem * pi2) @ (torch.eye(self.d, device=x.device) - self.M)
             + self.lam * x)
        return g  # (B,d)

    # ------------------------------------------------------------------ #
    #  Inner clock: K steps of gradient descent  (= thinking)            #
    # ------------------------------------------------------------------ #

    def descend(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        K: int = 8,
        step: float = 0.1,
        adaptive: bool = False,
    ) -> tuple[torch.Tensor, list[float]]:
        """
        Run K inner-clock steps.  Returns final x and energy trace.
        If adaptive=True, use learned halting (ACT-style).
        """
        energies = []
        for _ in range(K):
            g = self.grad_E(x, u)
            x = x - step * g
            energies.append(self.energy(x, u).mean().item())
            if adaptive:
                delta = torch.tensor([[energies[-2] - energies[-1]]]
                                     if len(energies) > 1 else [[1.0]],
                                     device=x.device)
                p_halt = torch.sigmoid(self.halt_net(delta))
                if p_halt.item() > 0.9:
                    break
        return x, energies

    # ------------------------------------------------------------------ #
    #  Fast-weight (test-time memory) update                              #
    # ------------------------------------------------------------------ #

    def update_M(
        self, x: torch.Tensor, eta: float = 0.01, gamma: float = 0.999
    ):
        """ΔM = η·(ε_mem·xᵀ) - γ·M  (Hebbian, applied in-place)."""
        with torch.no_grad():
            eps_mem = x - x @ self.M.T   # (B,d)
            # mean over batch
            self.M += eta * (eps_mem.T @ x) / x.shape[0] - gamma * self.M

    # ------------------------------------------------------------------ #
    #  Forward: one outer-clock tick                                      #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        K: int = 8,
        step: float = 0.1,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, list[float]]:
        x, energies = self.descend(x, u, K=K, step=step)
        if update_memory:
            self.update_M(x)
        return x, energies

    def reset_memory(self):
        self.M.zero_()
