"""
DEQ (Deep Equilibrium) 솔버.

Stage 1: Anderson acceleration으로 고정점 탐색.
Stage 2 (추후): Broyden + implicit differentiation (O(1) memory 학습).
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional


# ------------------------------------------------------------------ #
# Anderson Acceleration (fixed-point solver)                          #
# ------------------------------------------------------------------ #

def anderson_fixed_point(
    F: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    m: int = 6,
    lam: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-5,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Anderson acceleration으로 x* = F(x*) 고정점 탐색.

    F:        x → F(x) (한 스텝 맵)
    x0:       초기값 [B, d]
    m:        history window
    Returns:  (x*, info)
    """
    B, d = x0.shape
    X = torch.zeros(B, d, m, device=x0.device, dtype=x0.dtype)
    F_store = torch.zeros(B, d, m, device=x0.device, dtype=x0.dtype)

    x = x0.clone()
    f = F(x)

    X[..., 0] = x
    F_store[..., 0] = f
    x = f

    residuals = []
    info = {'converged': False, 'n_iter': 0}

    for k in range(1, max_iter):
        f = F(x)
        res = (f - x).norm().item()
        residuals.append(res)

        if res < tol:
            info['converged'] = True
            x = f
            break

        n = min(k, m)
        G = F_store[..., :n] - X[..., :n]  # [B, d, n]

        # Least-squares: min ||G c||  s.t. sum(c)=1
        GTG = torch.einsum('bdn,bdm->bnm', G, G)  # [B, n, n]
        GTG += lam * torch.eye(n, device=x0.device).unsqueeze(0)
        rhs = torch.ones(B, n, 1, device=x0.device, dtype=x0.dtype)
        try:
            c = torch.linalg.solve(GTG, rhs)          # [B, n, 1]
            c = c / c.sum(dim=1, keepdim=True).clamp(min=1e-8)
        except Exception:
            c = torch.ones(B, n, 1, device=x0.device) / n

        x_new = beta * torch.einsum('bdn,bn->bd', F_store[..., :n], c.squeeze(-1))
        x_new += (1 - beta) * torch.einsum('bdn,bn->bd', X[..., :n], c.squeeze(-1))

        idx = k % m
        X[..., idx] = x
        F_store[..., idx] = f
        x = x_new

    info['n_iter'] = k
    info['residuals'] = residuals
    return x, info


# ------------------------------------------------------------------ #
# DEQ backward (implicit differentiation)                             #
# ------------------------------------------------------------------ #

class _DEQBackward(torch.autograd.Function):
    """
    DEQ implicit differentiation.

    Forward:  x* = fixed-point (computed externally, passed in)
    Backward: (I − J_F)ᵀ v = ∂L/∂x*  →  ∂L/∂θ = v · ∂F/∂θ
    """

    @staticmethod
    def forward(ctx, x_star: torch.Tensor, F_fn, *params):
        ctx.save_for_backward(x_star)
        ctx.F_fn = F_fn
        return x_star

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_star, = ctx.saved_tensors
        F_fn = ctx.F_fn

        # Neumann 반복: v_{n+1} = grad + J_Fᵀ v_n
        v = grad_output.detach().clone()
        for _ in range(50):
            with torch.enable_grad():
                z = x_star.detach().requires_grad_(True)
                Fz = F_fn(z)
                JFT_v = torch.autograd.grad(Fz, z, v, retain_graph=False)[0]
            v_new = grad_output + JFT_v
            if (v_new - v).norm() < 1e-4 * (1 + v.norm()):
                break
            v = v_new

        # x_star 의 그래디언트 = v (체인룰 나머지는 autograd가 처리)
        return v, None, *([None] * len(ctx.saved_tensors[1:]))


class DEQSolver(nn.Module):
    """
    DEQ 레이어: 고정점 탐색 + implicit diff backward.

    forward():  Anderson acceleration으로 x* 탐색
    backward(): implicit differentiation으로 θ 업데이트
    """

    def __init__(
        self,
        max_iter: int = 50,
        tol: float = 1e-5,
        anderson_m: int = 6,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m

    def forward(
        self,
        F: Callable[[torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        params: Optional[list] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        x* = fixed_point(F, x0) + implicit diff backward.
        params: F의 파라미터 리스트 (backward를 위해).
        """
        with torch.no_grad():
            x_star, info = anderson_fixed_point(
                F, x0,
                m=self.anderson_m,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        if params is not None and any(p.requires_grad for p in params):
            x_star = _DEQBackward.apply(x_star.requires_grad_(True), F, *params)

        return x_star, info
