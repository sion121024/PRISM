"""
PRISM Emergence Suite — Kaggle GPU kernel (self-contained).

Goal: induce & measure EMERGENT abilities in PRISM along multiple axes:

  E1  Grokking          — training-time phase transition on modular arithmetic
                          (train acc saturates early, val acc jumps much later)
  E2  Scaling           — ability vs model width d (emergence over scale)
                          + "mirage check": exact-match acc vs continuous loss
  E3  Thinking depth K  — ability vs inner descent steps (test-time compute,
                          PRISM-unique axis), eval-K sweep + train-K sweep
  E4  Memory capacity   — associative recall: phase transition during training
                          + capacity heatmap (d x n_pairs), fast-weight eta trace
  E5  Weight decay      — regularization axis for grokking

Design notes vs repo v2 cell:
  * fast weight M is PER-SAMPLE (B,d,d) and DIFFERENTIABLE — the theory says
    dM = eta * eps_mem x^T - gamma*M; the repo's batch-mean update cannot store
    per-sequence associations, and without M a fully-converged x* depends only
    on the current token (memoryless). Differentiable M is the memory circuit
    whose formation we want to observe.
  * eta (memory write strength) is a learned scalar -> tracked over training as
    a mechanistic "circuit formation" signal.
  * no .item() inside the descent loop (CPU sync kills GPU throughput).

P100 note: recent torch wheels (cu12.8+) drop sm_60 kernels. bootstrap_torch()
detects a failing CUDA op and reinstalls torch==2.4.1+cu121, then re-execs.
"""

import json
import math
import os
import subprocess
import sys
import time

SMOKE = "--smoke" in sys.argv
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out"
os.makedirs(OUT_DIR, exist_ok=True)
MAX_HOURS = 8.0
T_START = time.time()


# --------------------------------------------------------------------------- #
#  0. torch / GPU bootstrap (handles P100 sm_60 vs new torch wheels)          #
# --------------------------------------------------------------------------- #

def bootstrap_torch():
    already_fixed = os.environ.get("PRISM_TORCH_FIXED") == "1"

    def cuda_works():
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            a = torch.randn(64, 64, device="cuda")
            _ = (a @ a).sum().item()
            print(f"[env] GPU OK: {torch.cuda.get_device_name(0)} "
                  f"cap={torch.cuda.get_device_capability(0)} "
                  f"torch={torch.__version__} cuda={torch.version.cuda}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[env] CUDA check failed: {type(e).__name__}: {e}")
            return False

    if cuda_works():
        return

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        gpus = smi.stdout.strip()
    except Exception:  # noqa: BLE001
        gpus = ""
    print(f"[env] nvidia-smi GPUs: {gpus!r}")

    if gpus and not already_fixed:
        # A physical GPU exists but torch can't use it -> arch mismatch
        # (e.g. P100 = sm_60, new wheels ship sm_70+ only). Install a wheel
        # that still carries sm_60 kernels.
        print("[env] reinstalling torch==2.4.1+cu121 for old GPU arch …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
             "--no-deps", "torch==2.4.1",
             "--index-url", "https://download.pytorch.org/whl/cu121"],
        )
        env = dict(os.environ, PRISM_TORCH_FIXED="1")
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    print("[env] no usable GPU — continuing on CPU with reduced budgets")


bootstrap_torch()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ON_GPU = DEVICE == "cuda"
if ON_GPU and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)


def hours_left() -> float:
    return MAX_HOURS - (time.time() - T_START) / 3600


# --------------------------------------------------------------------------- #
#  1. PRISM experimental cell + sequence model                                 #
# --------------------------------------------------------------------------- #

class FastWeights:
    """Per-sample fast weight M = sum_t c_t eps_t x_t^T, stored FACTORED.

    Never materializes (B,d,d): Mv / MTv are computed from the outer-product
    history in O(B*T*d). Exactly equivalent to the dense recurrence
    M <- (1-gamma) M + eta * eps x^T  (verified by fastweights_selftest).
    Enables d=512 models and large batches on 16GB GPUs.
    """

    def __init__(self):
        self.E = None   # (B, T, d) written eps vectors
        self.X = None   # (B, T, d) written x vectors
        self.c = None   # (B, T)    coefficients (decayed each write)

    def Mv(self, v):
        if self.E is None:
            return torch.zeros_like(v)
        s = torch.einsum("btd,bd->bt", self.X, v) * self.c
        return torch.einsum("bt,btd->bd", s, self.E)

    def MTv(self, v):
        if self.E is None:
            return torch.zeros_like(v)
        s = torch.einsum("btd,bd->bt", self.E, v) * self.c
        return torch.einsum("bt,btd->bd", s, self.X)

    def write(self, eps, x, eta, gamma):
        e1, x1 = eps.unsqueeze(1), x.unsqueeze(1)
        c1 = (eta * torch.ones(eps.size(0), 1, device=eps.device,
                               dtype=eps.dtype))
        if self.E is None:
            self.E, self.X, self.c = e1, x1, c1
        else:
            self.c = torch.cat([self.c * (1 - gamma), c1], dim=1)
            self.E = torch.cat([self.E, e1], dim=1)
            self.X = torch.cat([self.X, x1], dim=1)


class PRISMCellX(nn.Module):
    """Energy cell: E(x) = ½‖u−D·σ(x)‖²_Π1 + ½‖(I−M)x‖²_Π2 + ½λ‖x‖².

    nonlinear=True uses σ=tanh (round-2 fix). With a purely quadratic E,
    ∂E/∂x is linear in x, so the per-token state update is an affine map —
    the model cannot represent nonlinear tasks like modular arithmetic
    (round-1 result: transformer grokked, linear PRISM never even memorized).
    A nonlinear dictionary keeps the energy-descent principle intact while
    making x* a nonlinear function of the input.
    """

    def __init__(self, d: int, lam: float = 0.01, nonlinear: bool = True):
        super().__init__()
        self.d = d
        self.lam = lam
        self.nonlinear = nonlinear
        self.D = nn.Linear(d, d, bias=False)
        self.log_pi1 = nn.Parameter(torch.zeros(d))
        self.log_pi2 = nn.Parameter(torch.zeros(d))
        nn.init.orthogonal_(self.D.weight, gain=0.5)

    def grad_E(self, x, u, M):
        pi1 = self.log_pi1.exp()
        pi2 = self.log_pi2.exp()
        h = torch.tanh(x) if self.nonlinear else x
        eps_in = u - h @ self.D.weight.T
        g_in = -(eps_in * pi1) @ self.D.weight
        if self.nonlinear:
            g_in = g_in * (1 - h ** 2)  # σ'(x) chain rule
        if M is None:
            d_mem = x  # M = 0  =>  eps_mem = x,  (I−Mᵀ)eps = x
        else:
            eps_mem = x - M.Mv(x)
            d_mem = eps_mem - M.MTv(eps_mem)
        return g_in + d_mem * pi2 + self.lam * x

    def energy(self, x, u, M):
        pi1 = self.log_pi1.exp()
        pi2 = self.log_pi2.exp()
        h = torch.tanh(x) if self.nonlinear else x
        eps_in = u - h @ self.D.weight.T
        eps_mem = x if M is None else x - M.Mv(x)
        return (0.5 * (eps_in ** 2 * pi1).sum(-1)
                + 0.5 * (eps_mem ** 2 * pi2).sum(-1)
                + 0.5 * self.lam * (x ** 2).sum(-1))

    def descend(self, x, u, M, K: int, step: float, momentum: float):
        """Nesterov-accelerated inner descent. No host syncs inside the loop."""
        v = x
        for k in range(K):
            x_look = x + momentum * (x - v) if k > 0 else x
            g = self.grad_E(x_look, u, M)
            v, x = x, x_look - step * g
        return x


class PRISMSeq(nn.Module):
    """embed -> per token: K-step energy descent (+ differentiable Hebbian M) -> head."""

    def __init__(self, vocab: int, d: int, K: int = 8, step: float = 0.1,
                 momentum: float = 0.9, lam: float = 0.01,
                 fast_weights: bool = True, gamma: float = 0.05,
                 eta_init: float = 0.1, nonlinear: bool = True):
        super().__init__()
        self.d, self.K, self.step, self.momentum = d, K, step, momentum
        self.fast_weights = fast_weights
        self.gamma = gamma
        self.embed = nn.Embedding(vocab, d)
        self.cell = PRISMCellX(d, lam=lam, nonlinear=nonlinear)
        self.head = nn.Linear(d, vocab, bias=False)
        # learned memory-write strength, sigmoid-bounded to (0,1) — round-2
        # showed unbounded (softplus) eta ballooning to ~1.15 and
        # destabilizing training. Its trajectory is our circuit-formation
        # probe.
        self.eta_raw = nn.Parameter(
            torch.tensor(math.log(eta_init / (1.0 - eta_init))))
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    @property
    def eta(self):
        return torch.sigmoid(self.eta_raw)

    def forward(self, tokens, K: int | None = None):
        K = K or self.K
        B, T = tokens.shape
        x = torch.zeros(B, self.d, device=tokens.device)
        M = FastWeights() if self.fast_weights else None
        outs = []
        for t in range(T):
            u = self.embed(tokens[:, t])
            x = self.cell.descend(x, u, M, K, self.step, self.momentum)
            if self.fast_weights:
                eps = x - M.Mv(x)
                M.write(eps, x, self.eta, self.gamma)
            outs.append(x)
        h = torch.stack(outs, dim=1)
        return self.head(h)


class TinyTransformer(nn.Module):
    """2-layer causal transformer baseline for the grokking comparison."""

    def __init__(self, vocab: int, d: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, tokens, K=None):  # K ignored (API parity)
        B, T = tokens.shape
        h = self.embed(tokens) + self.pos(torch.arange(T, device=tokens.device))
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(h)


# --------------------------------------------------------------------------- #
#  2. Tasks                                                                    #
# --------------------------------------------------------------------------- #

def make_mod_dataset(p: int, op: str, train_frac: float, seed: int = 0):
    """[a, OP, b, EQ] -> c = a op b (mod p). vocab: 0..p-1, OP=p, EQ=p+1."""
    OP, EQ = p, p + 1
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if op == "add":
        c = (a + b) % p
    elif op == "mul":
        c = (a * b) % p
    elif op == "sub":
        c = (a - b) % p
    else:
        raise ValueError(op)
    x = torch.stack([a, torch.full_like(a, OP), b, torch.full_like(a, EQ)], 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=g)
    n_train = int(train_frac * p * p)
    tr, va = perm[:n_train], perm[n_train:]
    return (x[tr], c[tr]), (x[va], c[va]), p + 2


def recall_batch(batch: int, vocab: int, n_pairs: int, device,
                 generator=None):
    """[k1 v1 … kN vN SEP k_q] -> v_q.  SEP=0, symbols 1..vocab-1."""
    ks = torch.stack([
        torch.randperm(vocab - 1, generator=generator)[:n_pairs] + 1
        for _ in range(batch)])
    vs = torch.stack([
        torch.randperm(vocab - 1, generator=generator)[:n_pairs] + 1
        for _ in range(batch)])
    pairs = torch.stack([ks, vs], dim=2).reshape(batch, 2 * n_pairs)
    qi = torch.randint(0, n_pairs, (batch,), generator=generator)
    idx = torch.arange(batch)
    q = ks[idx, qi]
    y = vs[idx, qi]
    sep = torch.zeros(batch, 1, dtype=torch.long)
    x = torch.cat([pairs, sep, q.unsqueeze(1)], dim=1)
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
#  3. Train / eval loops                                                       #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_last(model, x, y, K=None, batch: int = 4096):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch):
        xb, yb = x[i:i + batch], y[i:i + batch]
        logits = model(xb, K=K)[:, -1]
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
    return correct / len(x), loss_sum / len(x)


def train_mod(model, train, val, *, max_steps, lr=1e-3, wd=1.0, batch=512,
              eval_every=100, tag="", early_acc=0.999, patience=3):
    """Full history of train/val acc+loss — the grokking curve."""
    xtr, ytr = (t.to(DEVICE) for t in train)
    xva, yva = (t.to(DEVICE) for t in val)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    hist = {"step": [], "train_acc": [], "train_loss": [],
            "val_acc": [], "val_loss": [], "eta": []}
    n = len(xtr)
    good, t0 = 0, time.time()
    for step_i in range(1, max_steps + 1):
        model.train()
        idx = torch.randint(0, n, (min(batch, n),), device=DEVICE)
        logits = model(xtr[idx])[:, -1]
        loss = F.cross_entropy(logits, ytr[idx])
        if not torch.isfinite(loss):
            print(f"[{tag}] DIVERGED at step {step_i}")
            hist["diverged_at"] = step_i
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step_i % eval_every == 0 or step_i == max_steps:
            tr_acc, tr_loss = eval_last(model, xtr, ytr)
            va_acc, va_loss = eval_last(model, xva, yva)
            eta = (model.eta.item() if hasattr(model, "eta_raw") else 0.0)
            for k, v in zip(
                ("step", "train_acc", "train_loss", "val_acc", "val_loss",
                 "eta"),
                    (step_i, tr_acc, tr_loss, va_acc, va_loss, eta)):
                hist[k].append(v)
            if step_i % (eval_every * 10) == 0 or step_i == max_steps:
                print(f"[{tag}] step {step_i:6d}  train {tr_acc:.3f}/"
                      f"{tr_loss:.4f}  val {va_acc:.3f}/{va_loss:.4f}  "
                      f"eta {eta:.3f}  ({time.time()-t0:.0f}s)")
            good = good + 1 if va_acc >= early_acc else 0
            if good >= patience:
                print(f"[{tag}] early stop at {step_i} (val acc "
                      f">= {early_acc})")
                break
    hist["wall_s"] = time.time() - t0
    return hist


def train_recall(model, *, vocab, n_pairs, max_steps, lr=3e-4, wd=0.01,
                 batch=256, eval_every=100, tag=""):
    g = torch.Generator().manual_seed(1234)
    xva, yva = recall_batch(2048, vocab, n_pairs, DEVICE, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.98))
    hist = {"step": [], "val_acc": [], "val_loss": [], "eta": []}
    t0 = time.time()
    for step_i in range(1, max_steps + 1):
        model.train()
        xb, yb = recall_batch(batch, vocab, n_pairs, DEVICE)
        logits = model(xb)[:, -1]
        loss = F.cross_entropy(logits, yb)
        if not torch.isfinite(loss):
            print(f"[{tag}] DIVERGED at step {step_i}")
            hist["diverged_at"] = step_i
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step_i % eval_every == 0 or step_i == max_steps:
            va_acc, va_loss = eval_last(model, xva, yva)
            eta = model.eta.item()
            hist["step"].append(step_i)
            hist["val_acc"].append(va_acc)
            hist["val_loss"].append(va_loss)
            hist["eta"].append(eta)
            if step_i % (eval_every * 10) == 0 or step_i == max_steps:
                print(f"[{tag}] step {step_i:6d}  val {va_acc:.3f}/"
                      f"{va_loss:.4f}  eta {eta:.3f}  "
                      f"({time.time()-t0:.0f}s)")
    hist["wall_s"] = time.time() - t0
    return hist


def transition_stats(steps, accs, lo=0.1, hi=0.9):
    """Where does acc cross lo/hi (emergence sharpness)?"""
    s_lo = next((s for s, a in zip(steps, accs) if a >= lo), None)
    s_hi = next((s for s, a in zip(steps, accs) if a >= hi), None)
    return {"step_10pct": s_lo, "step_90pct": s_hi,
            "transition_width": (s_hi - s_lo)
            if (s_hi is not None and s_lo is not None) else None,
            "final_acc": accs[-1] if accs else None}


# --------------------------------------------------------------------------- #
#  4. Experiment suite                                                         #
# --------------------------------------------------------------------------- #

RESULTS = {
    "env": {
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if ON_GPU else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if ON_GPU else None,
        "smoke": SMOKE,
    },
    "experiments": {},
}


def save_results():
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)


def sc(x, smoke_x):
    """scale a budget down in smoke mode"""
    return smoke_x if SMOKE else x


P = sc(97, 13)
GROK_STEPS = sc(50_000, 60)
SWEEP_STEPS = sc(10_000, 40)
RECALL_STEPS = sc(10_000, 40)
EVAL_EVERY = sc(250, 10)
D_MAIN = sc(128, 16)


CFG = {"lr": 1e-3, "nonlinear": True}


def fastweights_selftest():
    torch.manual_seed(3)
    B, d, eta, gamma = 5, 7, 0.3, 0.05
    fw = FastWeights()
    M = torch.zeros(B, d, d)
    for _ in range(6):
        x = torch.randn(B, d)
        eps = torch.randn(B, d)
        M = (1 - gamma) * M + eta * eps.unsqueeze(-1) * x.unsqueeze(1)
        fw.write(eps, x, torch.tensor(eta), gamma)
    v = torch.randn(B, d)
    dense = torch.einsum("bij,bj->bi", M, v)
    err = (dense - fw.Mv(v)).abs().max().item()
    errT = (torch.einsum("bji,bj->bi", M, v) - fw.MTv(v)).abs().max().item()
    assert err < 1e-4 and errT < 1e-4, (err, errT)
    print(f"[selftest] factored fast weights == dense (err {err:.2e})")


def lr_for(d: int) -> float:
    """Width-scaled lr: preflight tunes at d=64; larger widths need smaller
    steps (round-2: lr=1e-2 diverged at d>=256)."""
    return CFG["lr"] * min(1.0, (64.0 / d) ** 0.5)


def train_mod_retry(build, train, val, *, max_steps, wd, lr, eval_every,
                    tag, retries=2):
    """Divergence-safe training: halve lr and rebuild on NaN, up to N times."""
    for i in range(retries + 1):
        torch.manual_seed(42)
        m = build()
        h = train_mod(m, train, val, max_steps=max_steps, wd=wd, lr=lr,
                      eval_every=eval_every, tag=f"{tag} lr={lr:.2e}")
        if "diverged_at" not in h:
            return m, h, lr
        print(f"[{tag}] diverged -> retry with lr {lr/2:.2e}")
        lr /= 2
    return m, h, lr


def preflight():
    """Self-tuning: pick (nonlinear, lr) by short mod-23 memorization runs.
    Round-1 showed linear PRISM cannot even memorize mod arithmetic; this
    verifies the nonlinear fix on-device and picks a working lr."""
    print("\n=== preflight: nonlinearity x lr grid ===")
    train, val, vocab = make_mod_dataset(sc(23, 13), "add", 0.7, seed=1)
    out, best, best_acc = {}, None, -1.0
    for nonlin in (True, False):
        for lr in (1e-3, 3e-3, 1e-2):
            torch.manual_seed(7)
            m = PRISMSeq(vocab, d=64, K=8, nonlinear=nonlin).to(DEVICE)
            h = train_mod(m, train, val, max_steps=sc(1500, 30), wd=0.1,
                          lr=lr, eval_every=sc(50, 10),
                          tag=f"preflight nl={nonlin} lr={lr}",
                          early_acc=1.01)  # no early stop
            key = f"nl_{nonlin}_lr_{lr}"
            out[key] = {"train_acc": h["train_acc"][-1],
                        "val_acc": h["val_acc"][-1]}
            score = h["train_acc"][-1]
            if score > best_acc:
                best_acc, best = score, (nonlin, lr)
    RESULTS["experiments"]["preflight"] = out
    save_results()
    CFG["nonlinear"], CFG["lr"] = best
    print(f"preflight -> nonlinear={best[0]} lr={best[1]}  {out}")
    return True  # fast weights always on (theory requires memory via M)


def e1_grokking(use_fw: bool):
    """Flagship: modular arithmetic, 50% split, high wd, long training."""
    print("\n=== E1: grokking (training-time emergence) ===")
    exp = {}
    for op in ["add", "mul"]:
        if hours_left() < 2.0 and op == "mul":
            exp["mul"] = {"skipped": "time budget"}
            break
        train, val, vocab = make_mod_dataset(P, op, 0.5, seed=0)
        d1 = sc(512, 16)
        m, h, _ = train_mod_retry(
            lambda: PRISMSeq(vocab, d=d1, K=8, fast_weights=use_fw,
                             nonlinear=CFG["nonlinear"]).to(DEVICE),
            train, val, max_steps=GROK_STEPS, wd=1.0, lr=lr_for(d1),
            eval_every=EVAL_EVERY, tag=f"E1 {op}")
        h["params"] = sum(p.numel() for p in m.parameters())
        h["stats_train"] = transition_stats(h["step"], h["train_acc"])
        h["stats_val"] = transition_stats(h["step"], h["val_acc"])
        exp[op] = h
        torch.save(m.state_dict(), os.path.join(OUT_DIR, f"e1_{op}.pt"))
        RESULTS["experiments"]["e1_grokking"] = exp
        save_results()

    # transformer baseline on add (same budget)
    if hours_left() > 1.5:
        train, val, vocab = make_mod_dataset(P, "add", 0.5, seed=0)
        torch.manual_seed(42)
        tm = TinyTransformer(vocab, d=D_MAIN).to(DEVICE)
        h = train_mod(tm, train, val, max_steps=GROK_STEPS, wd=1.0,
                      eval_every=EVAL_EVERY, tag="E1 transformer add")
        h["params"] = sum(p.numel() for p in tm.parameters())
        h["stats_train"] = transition_stats(h["step"], h["train_acc"])
        h["stats_val"] = transition_stats(h["step"], h["val_acc"])
        exp["transformer_add"] = h
    RESULTS["experiments"]["e1_grokking"] = exp
    save_results()
    return exp


def e5_weight_decay(use_fw: bool):
    """wd axis: 0.0 / 0.1 / 1.0 — does emergence need regularization?"""
    print("\n=== E5: weight-decay axis ===")
    exp = {}
    train, val, vocab = make_mod_dataset(P, "add", 0.5, seed=0)
    for wd in [0.0, 0.1, 1.0]:
        if hours_left() < 1.2:
            exp[f"wd_{wd}"] = {"skipped": "time budget"}
            continue
        m, h, _ = train_mod_retry(
            lambda: PRISMSeq(vocab, d=D_MAIN, K=8, fast_weights=use_fw,
                             nonlinear=CFG["nonlinear"]).to(DEVICE),
            train, val, max_steps=sc(20_000, 40), wd=wd, lr=lr_for(D_MAIN),
            eval_every=EVAL_EVERY, tag=f"E5 wd={wd}")
        h["stats_val"] = transition_stats(h["step"], h["val_acc"])
        exp[f"wd_{wd}"] = h
        RESULTS["experiments"]["e5_weight_decay"] = exp
        save_results()
    return exp


def e2_scaling(use_fw: bool):
    """d sweep at fixed budget — emergence over scale + mirage check."""
    print("\n=== E2: scaling (width d) ===")
    exp = {}
    train, val, vocab = make_mod_dataset(P, "add", 0.5, seed=0)
    for d in sc([8, 16, 32, 64, 128, 256, 512], [8, 16]):
        if hours_left() < 0.8:
            exp[f"d_{d}"] = {"skipped": "time budget"}
            continue
        m, h, _ = train_mod_retry(
            lambda d=d: PRISMSeq(vocab, d=d, K=8, fast_weights=use_fw,
                                 nonlinear=CFG["nonlinear"]).to(DEVICE),
            train, val, max_steps=SWEEP_STEPS, wd=1.0, lr=lr_for(d),
            eval_every=EVAL_EVERY, tag=f"E2 d={d}")
        exp[f"d_{d}"] = {
            "params": sum(p.numel() for p in m.parameters()),
            "final_train_acc": h["train_acc"][-1] if h["train_acc"] else None,
            "final_val_acc": h["val_acc"][-1] if h["val_acc"] else None,
            "final_val_loss": h["val_loss"][-1] if h["val_loss"] else None,
            "stats_val": transition_stats(h["step"], h["val_acc"]),
            "curve_step": h["step"], "curve_val_acc": h["val_acc"],
            "curve_val_loss": h["val_loss"],
        }
        RESULTS["experiments"]["e2_scaling"] = exp
        save_results()
    return exp


def e3_thinking_depth(use_fw: bool):
    """K axis: eval-K sweep on the grokked model + train-K sweep."""
    print("\n=== E3: thinking depth K ===")
    exp = {"eval_K": {}, "train_K": {}}
    train, val, vocab = make_mod_dataset(P, "add", 0.5, seed=0)
    xva, yva = (t.to(DEVICE) for t in val)

    ckpt = os.path.join(OUT_DIR, "e1_add.pt")
    if os.path.exists(ckpt):
        m = PRISMSeq(vocab, d=sc(512, 16), K=8, fast_weights=use_fw,
                     nonlinear=CFG["nonlinear"]).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        for K in sc([1, 2, 4, 8, 16, 32, 64], [1, 4, 8]):
            acc, loss = eval_last(m, xva, yva, K=K)
            exp["eval_K"][f"K_{K}"] = {"val_acc": acc, "val_loss": loss}
            print(f"[E3 evalK] K={K:3d}  val_acc={acc:.3f}  loss={loss:.4f}")
    RESULTS["experiments"]["e3_thinking_depth"] = exp
    save_results()

    for K in sc([1, 2, 4, 8, 16], [1, 8]):
        if hours_left() < 0.5:
            exp["train_K"][f"K_{K}"] = {"skipped": "time budget"}
            continue
        m, h, _ = train_mod_retry(
            lambda K=K: PRISMSeq(vocab, d=D_MAIN, K=K, fast_weights=use_fw,
                                 nonlinear=CFG["nonlinear"]).to(DEVICE),
            train, val, max_steps=SWEEP_STEPS, wd=1.0, lr=lr_for(D_MAIN),
            eval_every=EVAL_EVERY, tag=f"E3 trainK={K}")
        exp["train_K"][f"K_{K}"] = {
            "final_train_acc": h["train_acc"][-1] if h["train_acc"] else None,
            "final_val_acc": h["val_acc"][-1] if h["val_acc"] else None,
            "stats_val": transition_stats(h["step"], h["val_acc"]),
            "curve_step": h["step"], "curve_val_acc": h["val_acc"],
        }
        RESULTS["experiments"]["e3_thinking_depth"] = exp
        save_results()
    return exp


def e4_memory(use_fw: bool):
    """Associative recall: training-time transition + capacity heatmap."""
    print("\n=== E4: associative recall (memory emergence) ===")
    vocab = 64
    exp = {"curves": {}, "capacity": {}}
    for n_pairs in sc([4, 8], [2]):
        if hours_left() < 0.6:
            exp["curves"][f"pairs_{n_pairs}"] = {"skipped": "time budget"}
            continue
        torch.manual_seed(42)
        m = PRISMSeq(vocab, d=D_MAIN, K=8, fast_weights=True, eta_init=0.2,
                     nonlinear=CFG["nonlinear"]).to(DEVICE)
        h = train_recall(m, vocab=vocab, n_pairs=n_pairs, lr=1e-3,
                         max_steps=RECALL_STEPS, eval_every=EVAL_EVERY,
                         tag=f"E4 pairs={n_pairs}")
        h["stats_val"] = transition_stats(h["step"], h["val_acc"])
        h["chance"] = 1.0 / (vocab - 1)
        exp["curves"][f"pairs_{n_pairs}"] = h
        RESULTS["experiments"]["e4_memory"] = exp
        save_results()

    # capacity heatmap d x n_pairs (short budget)
    for d in sc([32, 64, 128], [16]):
        for n_pairs in sc([2, 4, 8, 16], [2]):
            if hours_left() < 0.3:
                exp["capacity"][f"d{d}_p{n_pairs}"] = {"skipped": "time"}
                continue
            torch.manual_seed(42)
            m = PRISMSeq(vocab, d=d, K=8, fast_weights=True, eta_init=0.2,
                         nonlinear=CFG["nonlinear"]).to(DEVICE)
            h = train_recall(m, vocab=vocab, n_pairs=n_pairs, lr=1e-3,
                             max_steps=sc(5000, 20),
                             eval_every=sc(500, 10),
                             tag=f"E4cap d={d} p={n_pairs}")
            exp["capacity"][f"d{d}_p{n_pairs}"] = {
                "final_val_acc": h["val_acc"][-1] if h["val_acc"] else None,
                "final_eta": h["eta"][-1] if h["eta"] else None,
            }
            RESULTS["experiments"]["e4_memory"] = exp
            save_results()
    return exp


# --------------------------------------------------------------------------- #
#  5. Plots                                                                    #
# --------------------------------------------------------------------------- #

def make_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping plots")
        return

    ex = RESULTS["experiments"]

    if "e1_grokking" in ex:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, key in zip(axes, ["add", "mul"]):
            h = ex["e1_grokking"].get(key)
            if not h or "step" not in h:
                continue
            ax.plot(h["step"], h["train_acc"], label="train acc")
            ax.plot(h["step"], h["val_acc"], label="val acc")
            ax.set_xscale("log")
            ax.set_title(f"PRISM grokking — mod {key} (p={P})")
            ax.set_xlabel("step (log)")
            ax.set_ylabel("accuracy")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "e1_grokking.png"), dpi=120)
        plt.close(fig)

    if "e2_scaling" in ex:
        ds, accs, losses = [], [], []
        for k, v in sorted(ex["e2_scaling"].items(),
                           key=lambda kv: int(kv[0].split("_")[1])):
            if "final_val_acc" in v and v["final_val_acc"] is not None:
                ds.append(int(k.split("_")[1]))
                accs.append(v["final_val_acc"])
                losses.append(v["final_val_loss"])
        if ds:
            fig, ax1 = plt.subplots(figsize=(7, 4.5))
            ax1.plot(ds, accs, "o-", color="C0", label="val acc (exact match)")
            ax1.set_xscale("log", base=2)
            ax1.set_xlabel("width d (log2)")
            ax1.set_ylabel("val accuracy", color="C0")
            ax2 = ax1.twinx()
            ax2.plot(ds, losses, "s--", color="C3", label="val loss")
            ax2.set_ylabel("val loss (continuous)", color="C3")
            ax1.set_title("E2: emergence over scale + mirage check")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "e2_scaling.png"), dpi=120)
            plt.close(fig)

    if "e3_thinking_depth" in ex:
        ek = ex["e3_thinking_depth"].get("eval_K", {})
        if ek:
            Ks = sorted(int(k.split("_")[1]) for k in ek)
            accs = [ek[f"K_{K}"]["val_acc"] for K in Ks]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(Ks, accs, "o-")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("eval K — thinking depth (log2)")
            ax.set_ylabel("val accuracy")
            ax.set_title("E3: ability vs test-time compute (trained at K=8)")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "e3_evalK.png"), dpi=120)
            plt.close(fig)

    if "e4_memory" in ex:
        curves = ex["e4_memory"].get("curves", {})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = False
        for k, h in curves.items():
            if "step" in h:
                ax.plot(h["step"], h["val_acc"], label=k)
                plotted = True
        if plotted:
            ax.set_xlabel("step")
            ax.set_ylabel("val accuracy")
            ax.set_title("E4: associative recall — phase transition")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "e4_recall.png"), dpi=120)
        plt.close(fig)

    print("plots saved")


# --------------------------------------------------------------------------- #
#  6. Main                                                                     #
# --------------------------------------------------------------------------- #

def main():
    print(f"PRISM emergence suite — smoke={SMOKE} device={DEVICE} "
          f"p={P} d={D_MAIN}")
    fastweights_selftest()
    use_fw = preflight()
    RESULTS["use_fast_weights"] = use_fw
    RESULTS["config"] = dict(CFG)

    e1_grokking(use_fw)
    e4_memory(use_fw)          # memory axis early: unique PRISM claim
    e2_scaling(use_fw)
    e3_thinking_depth(use_fw)
    e5_weight_decay(use_fw)

    make_plots()
    save_results()
    print(f"\nDONE in {(time.time()-T_START)/60:.1f} min — results.json + "
          f"pngs in {OUT_DIR}")


if __name__ == "__main__":
    main()
