"""Improved causal time-slab PINN for the common heterogeneous wave benchmark.

Problem
-------
    u_tt - div(h(x,y) grad u) = 0,   (t,x,y) in (0,T) x (0,1)^2,
    u = 0 on the spatial boundary,
    u(0,x,y) = u0(x,y),
    u_t(0,x,y) = 0.

This version keeps the successful ingredients of the original PINN
(tanh MLP, normalized inputs, Sobol collocation, Adam -> L-BFGS), but fixes
three issues observed in the previous causal-slab calculation:

1. HARD STATE TRANSFER BETWEEN SLABS.
   For every slab [ta,tb], both displacement and velocity at t=ta are built
   directly into the trial solution. Hence u and u_t are continuous across
   slab interfaces up to floating-point/autodiff accuracy; no soft interface
   penalty is used.

2. PER-SLAB RESIDUAL NORMALIZATION.
   The PDE residual on each slab is normalized by the RMS magnitude of
   div(h grad u_left) at the left state of that slab, instead of using the
   very large initial-acceleration scale for all slabs.

3. RESIDUAL-ADAPTIVE COLLOCATION (RAR).
   During Adam training, high-residual points are periodically selected from
   a fresh candidate pool and retained in the collocation set. This follows
   the moving wavefront instead of oversampling only the old interface
   location.

4. DOUBLE PRECISION.
   float64 is used because the loss contains second derivatives of a narrow
   localized pulse.

The FDM reference is loaded only for evaluation and plotting, never for PINN
training.
"""
from __future__ import annotations

import csv
import gc
import os
import time
from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

# =============================================================================
# USER OPTIONS
# =============================================================================
FAST_TEST = False
MAKE_PLOTS = True
SEED = 1234
OUTPUT_DIR = "pinn_causal_slabs_v4_results"
REFERENCE_FILE = os.path.join("fdm_complex_results", "fdm_reference.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
T_FINAL = 0.75
SLAB_EDGES = (0.00, 0.25, 0.50, 0.75)
SNAPSHOT_TIMES = np.asarray(SLAB_EDGES, dtype=float)
GAUGE_THRESHOLD = 1.0e-2

# Same benchmark data as the FDM/FEM codes
H_D, H_S = 1.0, 0.35
X_S, W_S = 0.72, 0.04
DELTA_H_M = 0.18
X_M, Y_M, A_M, B_M = 0.56, 0.62, 0.09, 0.07
AMPLITUDE = 1.0
X0, Y0, A0, B0 = 0.24, 0.48, 0.055, 0.080
GAUGE_NAMES = ("G1", "G2", "G3", "G4")
GAUGES = np.array(
    [[0.38, 0.48], [0.56, 0.62], [0.69, 0.48], [0.84, 0.48]],
    dtype=float,
)


@dataclass(frozen=True)
class Config:
    # Network
    hidden_layers: int = 5
    width: int = 80

    # Adam
    adam_epochs: int = 3500
    pde_points: int = 5000
    left_pde_points: int = 1500
    learning_rate: float = 1.0e-3
    lr_step: int = 1750
    lr_gamma: float = 0.5
    grad_clip: float = 5.0

    # Residual-adaptive refinement (RAR)
    rar_every: int = 250
    rar_candidates: int = 12000
    rar_keep: int = 1200
    rar_batch: int = 1200

    # Relative weighting after per-slab normalization
    weight_pde: float = 1.0
    weight_left_pde: float = 2.0

    # L-BFGS
    lbfgs_points: int = 15000
    lbfgs_left_points: int = 3000
    lbfgs_iterations: int = 600
    lbfgs_history: int = 100

    # Scale estimation / evaluation
    scale_points: int = 6000
    eval_batch: int = 4096
    test_points: int = 12000
    test_batch: int = 1500
    energy_grid: int = 121
    energy_times: int = 41


CFG = Config()
if FAST_TEST:
    CFG = replace(
        CFG,
        hidden_layers=3,
        width=32,
        adam_epochs=8,
        pde_points=192,
        left_pde_points=64,
        lr_step=4,
        rar_every=4,
        rar_candidates=256,
        rar_keep=48,
        rar_batch=64,
        lbfgs_points=256,
        lbfgs_left_points=64,
        lbfgs_iterations=3,
        scale_points=128,
        eval_batch=512,
        test_points=192,
        test_batch=64,
        energy_grid=21,
        energy_times=5,
    )


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================
def coefficient_h(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    shelf = 0.5 * (H_D - H_S) * (1.0 - torch.tanh((x - X_S) / W_S))
    feature = DELTA_H_M * torch.exp(
        -((x - X_M) / A_M) ** 2 - ((y - Y_M) / B_M) ** 2
    )
    return H_S + shelf + feature


def boundary_factor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Zero on the boundary and max = 1 at (1/2,1/2).
    return 16.0 * x * (1.0 - x) * y * (1.0 - y)


def initial_displacement(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    denominator = X0 * (1.0 - X0) * Y0 * (1.0 - Y0)
    boundary = x * (1.0 - x) * y * (1.0 - y) / denominator
    gaussian = torch.exp(-((x - X0) / A0) ** 2 - ((y - Y0) / B0) ** 2)
    return AMPLITUDE * boundary * gaussian


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sobol(n: int, seed: int, dim: int = 3) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=int(seed))
    return engine.draw(n).to(device=DEVICE, dtype=DTYPE)


# =============================================================================
# NETWORK
# =============================================================================
class MLP(nn.Module):
    def __init__(self, hidden_layers: int, width: int, ta: float, tb: float):
        super().__init__()
        self.ta = float(ta)
        self.tb = float(tb)

        layers: list[nn.Module] = []
        in_features = 3
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Tanh())
            in_features = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(
                module.weight, gain=nn.init.calculate_gain("tanh")
            )
            nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_hat = 2.0 * (t - self.ta) / (self.tb - self.ta) - 1.0
        x_hat = 2.0 * x - 1.0
        y_hat = 2.0 * y - 1.0
        return self.net(torch.cat((t_hat, x_hat, y_hat), dim=1))


class SlabPINN(nn.Module):
    """PINN on one time slab with exact displacement/velocity transfer.

    On the first slab,
        u = u0 + s^2 B N.

    On later slabs,
        u = u_L + (t-ta) v_L + s^2 B N,

    where u_L and v_L are obtained from the frozen previous slab at t=ta.
    Therefore, at t=ta,
        u_new = u_old,   (u_new)_t = (u_old)_t
    exactly (up to autodiff/floating-point accuracy).
    """

    def __init__(
        self,
        ta: float,
        tb: float,
        config: Config,
        previous_model: Optional["SlabPINN"] = None,
    ):
        super().__init__()
        self.ta = float(ta)
        self.tb = float(tb)
        self.first_slab = previous_model is None
        self.net = MLP(config.hidden_layers, config.width, ta, tb)

        # Avoid registering the frozen previous model as a trainable submodule.
        object.__setattr__(self, "_previous_model", previous_model)
        if previous_model is not None:
            previous_model.eval()
            for p in previous_model.parameters():
                p.requires_grad_(False)

    @property
    def previous_model(self) -> Optional["SlabPINN"]:
        return object.__getattribute__(self, "_previous_model")

    def left_state(
        self, x: torch.Tensor, y: torch.Tensor, create_graph: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.first_slab:
            return initial_displacement(x, y), torch.zeros_like(x)

        previous = self.previous_model
        assert previous is not None
        t_left = torch.full_like(x, self.ta).detach().requires_grad_(True)
        u_left = previous(t_left, x, y)
        v_left = grad(
            u_left,
            t_left,
            torch.ones_like(u_left),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return u_left, v_left

    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = (t - self.ta) / (self.tb - self.ta)
        correction = s**2 * boundary_factor(x, y) * self.net(t, x, y)

        if self.first_slab:
            return initial_displacement(x, y) + correction

        # create_graph must remain enabled here because residual() differentiates
        # this transferred state in x and y.
        u_left, v_left = self.left_state(x, y, create_graph=True)
        return u_left + (t - self.ta) * v_left + correction


# =============================================================================
# DIFFERENTIAL OPERATORS
# =============================================================================
def residual(
    model: nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    u = model(t, x, y)
    one = torch.ones_like(u)

    u_t = grad(u, t, one, create_graph=True, retain_graph=True)[0]
    u_tt = grad(
        u_t,
        t,
        torch.ones_like(u_t),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    u_x = grad(u, x, one, create_graph=True, retain_graph=True)[0]
    u_y = grad(u, y, one, create_graph=True, retain_graph=True)[0]
    h = coefficient_h(x, y)

    div_x = grad(
        h * u_x,
        x,
        torch.ones_like(u_x),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    div_y = grad(
        h * u_y,
        y,
        torch.ones_like(u_y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return u_tt - div_x - div_y


def spatial_divergence_of_state(
    state_model: Optional[SlabPINN],
    t_value: float,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute div(h grad u_left) for per-slab scaling."""
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)

    if state_model is None:
        u = initial_displacement(x, y)
    else:
        t = torch.full_like(x, float(t_value))
        u = state_model(t, x, y)

    ux = grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    uy = grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    h = coefficient_h(x, y)
    divx = grad(h * ux, x, torch.ones_like(ux), create_graph=False, retain_graph=True)[0]
    divy = grad(h * uy, y, torch.ones_like(uy), create_graph=False)[0]
    return divx + divy


def slab_residual_scale(
    previous_model: Optional[SlabPINN], ta: float, n: int, seed: int
) -> float:
    """RMS initial acceleration scale for the current slab."""
    p = sobol(n, seed, dim=2)
    a = spatial_divergence_of_state(
        previous_model, ta, p[:, 0:1], p[:, 1:2]
    )
    rms = float(torch.sqrt(torch.mean(a.detach() ** 2)).cpu())
    return max(rms, 1.0e-3)


# =============================================================================
# SAMPLING
# =============================================================================
def spatial_source_points(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Half uniform and half concentrated around the narrow source."""
    p = sobol(n, seed, dim=2)
    x = p[:, 0:1].clone()
    y = p[:, 1:2].clone()
    m = n // 2

    xlo, xhi = max(0.0, X0 - 4.5 * A0), min(1.0, X0 + 4.5 * A0)
    ylo, yhi = max(0.0, Y0 - 4.5 * B0), min(1.0, Y0 + 4.5 * B0)
    x[:m] = xlo + (xhi - xlo) * p[:m, 0:1]
    y[:m] = ylo + (yhi - ylo) * p[:m, 1:2]
    return x, y


def sample_base_pde(
    n: int,
    seed: int,
    ta: float,
    tb: float,
    first_slab: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh Sobol points: 75% uniform + 25% left-time biased."""
    n_early = max(1, int(0.25 * n))
    n_uniform = max(1, n - n_early)

    pu = sobol(n_uniform, seed)
    pe = sobol(n_early, seed + 1)

    t1 = ta + (tb - ta) * pu[:, 0:1]
    x1 = pu[:, 1:2]
    y1 = pu[:, 2:3]

    t2 = ta + (tb - ta) * pe[:, 0:1] ** 2
    if first_slab:
        x2, y2 = spatial_source_points(n_early, seed + 2)
    else:
        x2, y2 = pe[:, 1:2], pe[:, 2:3]

    t = torch.cat((t1, t2), dim=0).detach().requires_grad_(True)
    x = torch.cat((x1, x2), dim=0).detach().requires_grad_(True)
    y = torch.cat((y1, y2), dim=0).detach().requires_grad_(True)
    return t, x, y


def sample_left_pde(
    n: int, seed: int, ta: float, first_slab: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if first_slab:
        x, y = spatial_source_points(n, seed)
    else:
        p = sobol(n, seed, dim=2)
        x, y = p[:, 0:1], p[:, 1:2]

    t = torch.full_like(x, float(ta)).detach().requires_grad_(True)
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    return t, x, y


def append_adaptive_points(
    base: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    adaptive: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if adaptive is None or adaptive.numel() == 0:
        return base

    t, x, y = base
    ta = adaptive[:, 0:1].detach()
    xa = adaptive[:, 1:2].detach()
    ya = adaptive[:, 2:3].detach()
    return (
        torch.cat((t.detach(), ta), dim=0).requires_grad_(True),
        torch.cat((x.detach(), xa), dim=0).requires_grad_(True),
        torch.cat((y.detach(), ya), dim=0).requires_grad_(True),
    )


def residual_scores_in_batches(
    model: SlabPINN,
    points: torch.Tensor,
    scale: float,
    batch_size: int,
) -> torch.Tensor:
    """Absolute normalized residual at candidate points without parameter backprop."""
    scores = []
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        chunk = points[start:end]
        t = chunk[:, 0:1].detach().requires_grad_(True)
        x = chunk[:, 1:2].detach().requires_grad_(True)
        y = chunk[:, 2:3].detach().requires_grad_(True)
        r = residual(model, t, x, y, create_graph=False)
        scores.append(torch.abs(r.detach() / scale).squeeze(1))
    return torch.cat(scores)


def select_rar_points(
    model: SlabPINN,
    ta: float,
    tb: float,
    scale: float,
    config: Config,
    seed: int,
) -> torch.Tensor:
    """Select the largest-residual space-time points from a fresh candidate pool."""
    p = sobol(config.rar_candidates, seed)
    candidates = p.clone()
    candidates[:, 0:1] = ta + (tb - ta) * candidates[:, 0:1]

    model.eval()
    with torch.enable_grad():
        scores = residual_scores_in_batches(
            model, candidates, scale, config.rar_batch
        )
    k = min(config.rar_keep, len(candidates))
    indices = torch.topk(scores, k=k, largest=True).indices
    selected = candidates[indices].detach()
    model.train()
    return selected


# =============================================================================
# INTERFACE DIAGNOSTICS
# =============================================================================
def evaluate_u_v(
    model: SlabPINN,
    t_value: float,
    x: torch.Tensor,
    y: torch.Tensor,
    create_graph: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.full_like(x, float(t_value)).detach().requires_grad_(True)
    xx = x.detach().requires_grad_(True)
    yy = y.detach().requires_grad_(True)
    u = model(t, xx, yy)
    v = grad(
        u,
        t,
        torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return u, v


# =============================================================================
# TRAIN ONE SLAB
# =============================================================================
def train_slab(
    index: int,
    ta: float,
    tb: float,
    config: Config,
    previous_model: Optional[SlabPINN] = None,
) -> tuple[SlabPINN, dict]:
    first = previous_model is None
    set_seed(SEED + 1000 * index)

    # Scale is computed BEFORE constructing the new slab, from its inherited
    # left displacement field.
    scale = slab_residual_scale(
        previous_model,
        ta,
        config.scale_points,
        SEED + 50000 + index,
    )
    print(f"[slab {index+1}] residual scale = {scale:.6e}")

    model = SlabPINN(ta, tb, config, previous_model=previous_model).to(
        device=DEVICE, dtype=DTYPE
    )

    # Weight transfer is useful only for the trainable correction network.
    if previous_model is not None:
        model.net.load_state_dict(previous_model.net.state_dict())

    optimizer = torch.optim.Adam(model.net.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step, gamma=config.lr_gamma
    )

    adaptive_points: Optional[torch.Tensor] = None
    adam_history: list[list[float]] = []
    rar_history: list[tuple[int, float, float]] = []

    tic = time.perf_counter()
    for epoch in range(1, config.adam_epochs + 1):
        # Refresh residual-adaptive points periodically. Do it after the first
        # few Adam steps so the initial network is not used to define them.
        if (
            config.rar_every > 0
            and epoch > 1
            and (epoch - 1) % config.rar_every == 0
        ):
            adaptive_points = select_rar_points(
                model,
                ta,
                tb,
                scale,
                config,
                SEED + 400000 * index + epoch,
            )
            with torch.enable_grad():
                scores = residual_scores_in_batches(
                    model, adaptive_points, scale, config.rar_batch
                )
            rar_history.append(
                (
                    epoch,
                    float(torch.mean(scores).cpu()),
                    float(torch.max(scores).cpu()),
                )
            )
            print(
                f"[slab {index+1} RAR] epoch={epoch:4d}: "
                f"mean|R|/S={rar_history[-1][1]:.3e}, "
                f"max|R|/S={rar_history[-1][2]:.3e}"
            )

        n_base = max(64, config.pde_points - (0 if adaptive_points is None else len(adaptive_points)))
        base = sample_base_pde(
            n_base,
            SEED + 100000 * index + epoch,
            ta,
            tb,
            first_slab=first,
        )
        t, x, y = append_adaptive_points(base, adaptive_points)
        tl, xl, yl = sample_left_pde(
            config.left_pde_points,
            SEED + 300000 * index + epoch,
            ta,
            first_slab=first,
        )

        optimizer.zero_grad(set_to_none=True)
        rpde = residual(model, t, x, y, create_graph=True) / scale
        rleft = residual(model, tl, xl, yl, create_graph=True) / scale
        lpde = torch.mean(rpde**2)
        lleft = torch.mean(rleft**2)
        loss = config.weight_pde * lpde + config.weight_left_pde * lleft
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.net.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        adam_history.append(
            [
                float(loss.detach().cpu()),
                float(lpde.detach().cpu()),
                float(lleft.detach().cpu()),
            ]
        )

        if epoch % 500 == 0 or epoch == config.adam_epochs:
            h = adam_history[-1]
            print(
                f"[slab {index+1} Adam] {epoch:4d}/{config.adam_epochs}: "
                f"total={h[0]:.3e}, PDE={h[1]:.3e}, leftPDE={h[2]:.3e}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    adam_time = time.perf_counter() - tic

    # ---------------- L-BFGS ----------------
    # Freeze the final Adam RAR points into the fixed L-BFGS set.
    n_base = max(128, config.lbfgs_points - (0 if adaptive_points is None else len(adaptive_points)))
    base = sample_base_pde(
        n_base,
        SEED + 700000 + index,
        ta,
        tb,
        first_slab=first,
    )
    t, x, y = append_adaptive_points(base, adaptive_points)
    tl, xl, yl = sample_left_pde(
        config.lbfgs_left_points,
        SEED + 710000 + index,
        ta,
        first_slab=first,
    )

    optimizer2 = torch.optim.LBFGS(
        model.net.parameters(),
        max_iter=config.lbfgs_iterations,
        history_size=config.lbfgs_history,
        tolerance_grad=1.0e-9,
        tolerance_change=1.0e-12,
        line_search_fn="strong_wolfe",
    )
    lbfgs_history: list[list[float]] = []

    def closure():
        optimizer2.zero_grad(set_to_none=True)
        lpde = torch.mean((residual(model, t, x, y, True) / scale) ** 2)
        lleft = torch.mean((residual(model, tl, xl, yl, True) / scale) ** 2)
        loss = config.weight_pde * lpde + config.weight_left_pde * lleft
        loss.backward()
        lbfgs_history.append(
            [
                float(loss.detach().cpu()),
                float(lpde.detach().cpu()),
                float(lleft.detach().cpu()),
            ]
        )
        return loss

    print(f"[slab {index+1}] L-BFGS refinement ...")
    tic = time.perf_counter()
    optimizer2.step(closure)
    lbfgs_time = time.perf_counter() - tic

    return model, {
        "residual_scale": scale,
        "adam_time_s": adam_time,
        "lbfgs_time_s": lbfgs_time,
        "training_time_s": adam_time + lbfgs_time,
        "adam_history": np.asarray(adam_history),
        "lbfgs_history": np.asarray(lbfgs_history),
        "rar_history": np.asarray(rar_history, dtype=float),
        "parameters": sum(p.numel() for p in model.net.parameters()),
    }


# =============================================================================
# MULTI-SLAB EVALUATION
# =============================================================================
def slab_index(t_value: float) -> int:
    if t_value <= SLAB_EDGES[1] + 1.0e-12:
        return 0
    if t_value <= SLAB_EDGES[2] + 1.0e-12:
        return 1
    return 2


def predict_field(
    models: list[SlabPINN],
    t_value: float,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    gradients: bool = False,
):
    model = models[slab_index(float(t_value))]
    xs, ys = X.ravel(), Y.ravel()
    u_parts, ux_parts, uy_parts = [], [], []

    for start in range(0, len(xs), batch_size):
        end = min(start + batch_size, len(xs))
        t = torch.full(
            (end - start, 1), float(t_value), device=DEVICE, dtype=DTYPE
        )
        x = torch.as_tensor(xs[start:end, None], device=DEVICE, dtype=DTYPE)
        y = torch.as_tensor(ys[start:end, None], device=DEVICE, dtype=DTYPE)

        if not gradients:
            # Do not use torch.no_grad() here: later-slab forward passes need
            # autodiff internally to construct the inherited velocity field.
            with torch.enable_grad():
                u = model(t, x, y)
            u_parts.append(u.detach().cpu().numpy().ravel())
            continue

        x.requires_grad_(True)
        y.requires_grad_(True)
        u = model(t, x, y)
        ux = grad(u, x, torch.ones_like(u), create_graph=False, retain_graph=True)[0]
        uy = grad(u, y, torch.ones_like(u), create_graph=False)[0]
        u_parts.append(u.detach().cpu().numpy().ravel())
        ux_parts.append(ux.detach().cpu().numpy().ravel())
        uy_parts.append(uy.detach().cpu().numpy().ravel())

    shape = X.shape
    u = np.concatenate(u_parts).reshape(shape)
    if gradients:
        return (
            u,
            np.concatenate(ux_parts).reshape(shape),
            np.concatenate(uy_parts).reshape(shape),
        )
    return u


def predict_velocity(
    models: list[SlabPINN],
    t_value: float,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model = models[slab_index(float(t_value))]
    xs, ys = X.ravel(), Y.ravel()
    out = []
    for start in range(0, len(xs), batch_size):
        end = min(start + batch_size, len(xs))
        t = torch.full(
            (end - start, 1),
            float(t_value),
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        x = torch.as_tensor(xs[start:end, None], device=DEVICE, dtype=DTYPE)
        y = torch.as_tensor(ys[start:end, None], device=DEVICE, dtype=DTYPE)
        u = model(t, x, y)
        v = grad(u, t, torch.ones_like(u), create_graph=False)[0]
        out.append(v.detach().cpu().numpy().ravel())
    return np.concatenate(out).reshape(X.shape)


def integrate2d(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(np.trapezoid(values, y, axis=1), x, axis=0))


def arrival_time(times: np.ndarray, signal: np.ndarray) -> float:
    z = np.abs(signal)
    ids = np.flatnonzero(z >= GAUGE_THRESHOLD)
    if len(ids) == 0:
        return np.nan
    k = int(ids[0])
    if k == 0 or z[k] <= z[k - 1]:
        return float(times[k])
    alpha = (GAUGE_THRESHOLD - z[k - 1]) / (z[k] - z[k - 1])
    return float(times[k - 1] + alpha * (times[k] - times[k - 1]))


def evaluate(
    models: list[SlabPINN],
    reference,
    config: Config,
    slab_scales: list[float],
) -> dict:
    x = reference["common_x"]
    y = reference["common_y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    ref = reference["final_field"]
    refux, refuy = reference["final_ux"], reference["final_uy"]

    u, ux, uy = predict_field(
        models, T_FINAL, X, Y, config.eval_batch, gradients=True
    )
    d = u - ref
    rel_l2 = np.sqrt(
        integrate2d(d * d, x, y) / max(integrate2d(ref * ref, x, y), 1.0e-30)
    )
    linf = float(np.max(np.abs(d)))
    rel_h1 = np.sqrt(
        integrate2d(d * d + (ux - refux) ** 2 + (uy - refuy) ** 2, x, y)
        / max(
            integrate2d(ref * ref + refux**2 + refuy**2, x, y),
            1.0e-30,
        )
    )

    # Gauge records
    times = reference["gauge_times"]
    ref_g = reference["gauge_values"]
    pred_g = np.zeros_like(ref_g)
    gx_np = GAUGES[:, 0]
    gy_np = GAUGES[:, 1]
    GX, GY = gx_np.reshape(-1, 1), gy_np.reshape(-1, 1)

    for j, tv in enumerate(times):
        model = models[slab_index(float(tv))]
        t = torch.full((len(GAUGES), 1), float(tv), device=DEVICE, dtype=DTYPE)
        gx = torch.as_tensor(GX, device=DEVICE, dtype=DTYPE)
        gy = torch.as_tensor(GY, device=DEVICE, dtype=DTYPE)
        with torch.enable_grad():
            pred_g[j] = model(t, gx, gy).detach().cpu().numpy().ravel()

    gauge_errors, arrival_errors = [], []
    for k in range(len(GAUGES)):
        num = np.trapezoid((pred_g[:, k] - ref_g[:, k]) ** 2, times)
        den = np.trapezoid(ref_g[:, k] ** 2, times)
        gauge_errors.append(np.sqrt(num / max(den, 1.0e-30)))
        a_pred = arrival_time(times, pred_g[:, k])
        a_ref = arrival_time(times, ref_g[:, k])
        arrival_errors.append(abs(a_pred - a_ref) if np.isfinite(a_pred) and np.isfinite(a_ref) else np.nan)

    # Independent normalized residual RMSE, separately normalized on each slab
    residual_mse = []
    per_slab = max(300, config.test_points // 3)
    for i, (ta, tb) in enumerate(zip(SLAB_EDGES[:-1], SLAB_EDGES[1:])):
        base = sample_base_pde(
            per_slab,
            SEED + 900000 + i,
            ta,
            tb,
            first_slab=(i == 0),
        )
        t, xx, yy = base
        total_sq = 0.0
        count = 0
        for start in range(0, len(t), config.test_batch):
            end = min(start + config.test_batch, len(t))
            r = residual(
                models[i],
                t[start:end],
                xx[start:end],
                yy[start:end],
                create_graph=False,
            ) / slab_scales[i]
            total_sq += float(torch.sum(r.detach() ** 2).cpu())
            count += len(r)
        residual_mse.append(total_sq / max(count, 1))
    residual_rmse = float(np.sqrt(np.mean(residual_mse)))

    # Energy curve
    xe = np.linspace(0.0, 1.0, config.energy_grid)
    ye = np.linspace(0.0, 1.0, config.energy_grid)
    XE, YE = np.meshgrid(xe, ye, indexing="ij")
    Hnp = (
        H_S
        + 0.5 * (H_D - H_S) * (1.0 - np.tanh((XE - X_S) / W_S))
        + DELTA_H_M
        * np.exp(-((XE - X_M) / A_M) ** 2 - ((YE - Y_M) / B_M) ** 2)
    )
    energy_times = np.linspace(0.0, T_FINAL, config.energy_times)
    energies = []
    for tv in energy_times:
        _, uxx, uyy = predict_field(
            models, float(tv), XE, YE, config.eval_batch, gradients=True
        )
        vv = predict_velocity(models, float(tv), XE, YE, config.eval_batch)
        energies.append(
            0.5 * integrate2d(vv * vv + Hnp * (uxx * uxx + uyy * uyy), xe, ye)
        )
    energies = np.asarray(energies)
    energy_drift = float(
        np.max(np.abs(energies - energies[0])) / max(abs(energies[0]), 1.0e-30)
    )

    snapshots = {
        float(tv): predict_field(
            models, float(tv), X, Y, config.eval_batch, gradients=False
        )
        for tv in SNAPSHOT_TIMES
    }

    # Interface jumps should now be essentially roundoff-level because both u
    # and u_t are imposed hard in the later-slab ansatz.
    interface_jumps = []
    n_jump = 256 if FAST_TEST else 5000
    for k, tv in enumerate(SLAB_EDGES[1:-1], start=1):
        pts = sobol(n_jump, SEED + 950000 + k, dim=2)
        xx, yy = pts[:, 0:1], pts[:, 1:2]
        with torch.enable_grad():
            up, vp = evaluate_u_v(models[k - 1], tv, xx, yy)
            uc, vc = evaluate_u_v(models[k], tv, xx, yy)
        jump_u = float(torch.sqrt(torch.mean((up.detach() - uc.detach()) ** 2)).cpu())
        jump_v = float(torch.sqrt(torch.mean((vp.detach() - vc.detach()) ** 2)).cpu())
        interface_jumps.append((jump_u, jump_v))

    return {
        "relative_L2": float(rel_l2),
        "Linf": linf,
        "relative_H1": float(rel_h1),
        "residual_RMSE": residual_rmse,
        "mean_gauge_error": float(np.mean(gauge_errors)),
        "max_arrival_error": float(np.nanmax(arrival_errors)),
        "energy_drift": energy_drift,
        "interface_jumps": interface_jumps,
        "x": x,
        "y": y,
        "reference_field": ref,
        "predicted_field": u,
        "gauge_times": times,
        "reference_gauges": ref_g,
        "predicted_gauges": pred_g,
        "energy_times": energy_times,
        "energies": energies,
        "snapshots": snapshots,
    }


# =============================================================================
# PLOTS
# =============================================================================
def make_plots(metrics: dict, training_info: list[dict]) -> None:
    X, Y = np.meshgrid(metrics["x"], metrics["y"], indexing="ij")

    # Wavefield snapshots, same format as FDM/FEM
    fig = plt.figure(figsize=(18, 4.8))
    for k, tv in enumerate(SNAPSHOT_TIMES):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        field = metrics["snapshots"][float(tv)]
        surf = ax.plot_surface(
            X,
            Y,
            field,
            cmap="viridis",
            edgecolor="black",
            linewidth=0.1,
            rstride=2,
            cstride=2,
            alpha=0.95,
        )
        ax.set_title(fr"$t={tv:.2f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$u(t,x,y)$")
        ax.view_init(elev=30, azim=110)
        fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.10)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "pinn_wavefield_snapshots_causal_v4.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Final field comparison
    fig = plt.figure(figsize=(18, 5))
    panels = [
        (metrics["reference_field"], "FDM reference at T", "viridis"),
        (metrics["predicted_field"], "Causal PINN v4 at T", "viridis"),
        (
            np.abs(metrics["predicted_field"] - metrics["reference_field"]),
            "Absolute error",
            "inferno",
        ),
    ]
    for j, (field, title, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        ax.plot_surface(X, Y, field, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_final_field_causal_v4.png"), dpi=300)
    plt.close(fig)

    # Gauges
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        ax.plot(
            metrics["gauge_times"],
            metrics["reference_gauges"][:, k],
            label="FDM reference",
        )
        ax.plot(
            metrics["gauge_times"],
            metrics["predicted_gauges"][:, k],
            "--",
            label="Causal PINN v4",
        )
        ax.set_title(GAUGE_NAMES[k])
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_gauge_records_causal_v4.png"), dpi=300)
    plt.close(fig)

    # Energy
    rel_e = (metrics["energies"] - metrics["energies"][0]) / metrics["energies"][0]
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(metrics["energy_times"], rel_e)
    plt.xlabel("t")
    plt.ylabel(r"$(E_\theta(t)-E_\theta(0))/E_\theta(0)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_energy_causal_v4.png"), dpi=300)
    plt.close()

    # Training losses
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (ax, info) in enumerate(zip(axes, training_info)):
        h = info["adam_history"]
        ax.semilogy(h[:, 1], label="PDE")
        ax.semilogy(h[:, 2], label="left PDE")
        ax.set_title(
            f"Slab {i+1}: [{SLAB_EDGES[i]:.2f},{SLAB_EDGES[i+1]:.2f}]\n"
            f"scale={info['residual_scale']:.2e}"
        )
        ax.set_xlabel("Adam epoch")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_training_causal_v4.png"), dpi=300)
    plt.close(fig)

    # RAR diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for i, (ax, info) in enumerate(zip(axes, training_info)):
        rh = info["rar_history"]
        if len(rh) > 0:
            ax.semilogy(rh[:, 0], rh[:, 1], "o-", label="mean selected residual")
            ax.semilogy(rh[:, 0], rh[:, 2], "s-", label="max selected residual")
        ax.set_title(f"Slab {i+1} RAR")
        ax.set_xlabel("Adam epoch")
        ax.set_ylabel(r"normalized $|R|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pinn_rar_causal_v4.png"), dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("Improved causal time-slab PINN (v4)")
    print(f"Device: {DEVICE}")
    print(f"dtype:  {DTYPE}")
    print(f"FAST_TEST={FAST_TEST}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not os.path.exists(REFERENCE_FILE):
        raise FileNotFoundError(
            f"Missing {REFERENCE_FILE}. Run TsunamiEquation_ComplexCase_FDM.py first."
        )

    reference = np.load(REFERENCE_FILE, allow_pickle=False)

    models: list[SlabPINN] = []
    infos: list[dict] = []
    previous: Optional[SlabPINN] = None

    for i, (ta, tb) in enumerate(zip(SLAB_EDGES[:-1], SLAB_EDGES[1:])):
        print(f"\n=== Slab {i+1}: [{ta:.2f}, {tb:.2f}] ===")
        model, info = train_slab(
            i,
            ta,
            tb,
            CFG,
            previous_model=previous,
        )
        models.append(model)
        infos.append(info)
        previous = model

        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    slab_scales = [info["residual_scale"] for info in infos]
    metrics = evaluate(models, reference, CFG, slab_scales)
    total_time = sum(info["training_time_s"] for info in infos)
    total_parameters = sum(info["parameters"] for info in infos)

    print("\n=== FINAL DIAGNOSTICS ===")
    print("Per-slab residual scales:")
    for i, scale in enumerate(slab_scales, start=1):
        print(f"  slab {i}: {scale:.6e}")
    print(f"relative L2                 = {metrics['relative_L2']:.6e}")
    print(f"Linf                        = {metrics['Linf']:.6e}")
    print(f"relative H1                 = {metrics['relative_H1']:.6e}")
    print(f"normalized residual RMSE    = {metrics['residual_RMSE']:.6e}")
    print(f"mean gauge error            = {metrics['mean_gauge_error']:.6e}")
    print(f"maximum arrival-time error  = {metrics['max_arrival_error']:.6e}")
    print(f"energy drift                = {metrics['energy_drift']:.6e}")
    for i, (ju, jv) in enumerate(metrics["interface_jumps"], start=1):
        print(
            f"interface t={SLAB_EDGES[i]:.2f}: "
            f"u RMS jump={ju:.3e}, v RMS jump={jv:.3e}"
        )
    print(f"total training time         = {total_time:.3f} s")
    print(f"trainable parameters/slab   = {[info['parameters'] for info in infos]}")
    print(f"parameters across 3 slabs   = {total_parameters}")

    row = {
        "method": "PINN-causal-slabs-v4",
        "seed": SEED,
        "dtype": str(DTYPE),
        "slabs": 3,
        "parameters_total": total_parameters,
        "scale_slab1": slab_scales[0],
        "scale_slab2": slab_scales[1],
        "scale_slab3": slab_scales[2],
        "relative_L2": metrics["relative_L2"],
        "Linf": metrics["Linf"],
        "relative_H1": metrics["relative_H1"],
        "normalized_residual_RMSE": metrics["residual_RMSE"],
        "mean_gauge_error": metrics["mean_gauge_error"],
        "maximum_arrival_time_error": metrics["max_arrival_error"],
        "energy_drift": metrics["energy_drift"],
        "interface_u_jump_t025": metrics["interface_jumps"][0][0],
        "interface_v_jump_t025": metrics["interface_jumps"][0][1],
        "interface_u_jump_t050": metrics["interface_jumps"][1][0],
        "interface_v_jump_t050": metrics["interface_jumps"][1][1],
        "total_training_time_s": total_time,
    }

    csv_path = os.path.join(OUTPUT_DIR, "pinn_causal_slabs_v4_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved: {csv_path}")

    if MAKE_PLOTS:
        make_plots(metrics, infos)


if __name__ == "__main__":
    main()