# ============================================================
# PINN for u_tt - div(H grad u) = f on [0,T] x [0,1]^2
#
# This code:
#   1. reproducible train/evaluate functions,
#   2. final-time L2, relative L2, Linfinity and relative H1 errors,
#   3. independent space-time solution and residual errors,
#   4. relative energy errors,
#   5. separate Adam and L-BFGS loss plots,
#   6. optional five-seed baseline statistics,
#   7. optional one-factor-at-a-time hyperparameter sensitivity study,
#   8. CSV files that can be copied into the manuscript tables.
#
# The ordinary baseline run still produces the same three main figures:
#   - exact/PINN/error surfaces,
#   - PINN/exact energy,
#   - Adam and L-BFGS loss histories.
# ============================================================

import csv
import gc
import os
import timeit
import tracemalloc
from dataclasses import dataclass, replace

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad


# ============================================================
# USER OPTIONS
# ============================================================

# Ordinary baseline run with the original plots.
RUN_BASELINE_PLOTS = False

# Five independent baseline runs for Table: baseline diagnostics.
# This is computationally expensive; enable it for the final paper results.
RUN_BASELINE_STATISTICS = False

# One-factor-at-a-time sensitivity study for Table: sensitivity results.
# With five seeds and ten configurations this performs 50 training runs.
RUN_SENSITIVITY_STUDY = True

# Set True only to verify the program quickly.
# Never use FAST_TEST results in the manuscript.
FAST_TEST = False

OUTPUT_DIR = "pinn_results_simple"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASELINE_SEED = 1234
STATISTICAL_SEEDS = [1234, 2345, 3456, 4567, 5678]
T_final = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    name: str = "Baseline"
    hidden_layers: int = 4
    width: int = 64
    dtype_name: str = "float32"

    epochs_adam: int = 3000
    N_pde_adam: int = 3000
    learning_rate: float = 2.0e-3
    lr_step: int = 1500
    lr_gamma: float = 0.5
    grad_clip: float = 1.0

    N_pde_lbfgs: int = 10000
    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1.0e-9
    lbfgs_tolerance_change: float = 1.0e-12

    n_final: int = 101
    n_energy: int = 129
    n_energy_times: int = 41
    N_test: int = 10000


def get_dtype(config):
    if config.dtype_name == "float64":
        return torch.float64
    if config.dtype_name == "float32":
        return torch.float32
    raise ValueError("dtype_name must be 'float32' or 'float64'.")


BASELINE = Config()

if FAST_TEST:
    BASELINE = replace(
        BASELINE,
        epochs_adam=5,
        N_pde_adam=64,
        lr_step=3,
        N_pde_lbfgs=128,
        lbfgs_max_iter=3,
        n_final=21,
        n_energy=21,
        n_energy_times=5,
        N_test=128,
    )


# ============================================================
# EXACT DATA AND COEFFICIENTS
# ============================================================

def phi(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def exact_u(t, x, y):
    return torch.cos(t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def exact_ux(t, x, y):
    return torch.pi * torch.cos(t) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)


def exact_uy(t, x, y):
    return torch.pi * torch.cos(t) * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)


def H(x, y):
    return 1.0 + x**2 + y**2


def f(t, x, y):
    sxs = torch.sin(torch.pi * x)
    cxs = torch.cos(torch.pi * x)
    sys = torch.sin(torch.pi * y)
    cys = torch.cos(torch.pi * y)

    term1 = sxs * sys
    term2 = 2.0 * torch.pi * x * cxs * sys
    term3 = 2.0 * torch.pi * y * sxs * cys
    term4 = 2.0 * torch.pi**2 * (1.0 + x**2 + y**2) * sxs * sys
    return -torch.cos(t) * (term1 + term2 + term3 - term4)


def exact_energy_np(t):
    """Exact energy: E(t)=1/8 sin^2(t)+5 pi^2/12 cos^2(t)."""
    t = np.asarray(t, dtype=float)
    return 0.125 * np.sin(t) ** 2 + (5.0 * np.pi**2 / 12.0) * np.cos(t) ** 2


# ============================================================
# NETWORK AND HARD-CONSTRAINED TRIAL SOLUTION
# ============================================================

class MLP(nn.Module):
    def __init__(self, hidden_layers=4, width=64, T=T_final):
        super().__init__()
        self.T = T

        layers = []
        input_dim = 3
        for layer_index in range(hidden_layers):
            layers.append(nn.Linear(input_dim if layer_index == 0 else width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(
                module.weight,
                gain=nn.init.calculate_gain("tanh"),
            )
            nn.init.zeros_(module.bias)

    def forward(self, t, x, y):
        t_hat = 2.0 * t / self.T - 1.0
        x_hat = 2.0 * x - 1.0
        y_hat = 2.0 * y - 1.0
        inputs = torch.cat([t_hat, x_hat, y_hat], dim=1)
        return self.net(inputs)


class PINNTrial(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self.net = base_net

    def forward(self, t, x, y):
        multiplier = t**2 * x * (1.0 - x) * y * (1.0 - y)
        return phi(x, y) + multiplier * self.net(t, x, y)


# ============================================================
# SAMPLING
# ============================================================

def sample_collocation_sobol(N, dtype, device, seed, T=T_final):
    engine = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=int(seed),
    )
    points = engine.draw(N).to(device=device, dtype=dtype)

    t = (points[:, 0:1] * T).detach().requires_grad_(True)
    x = points[:, 1:2].detach().requires_grad_(True)
    y = points[:, 2:3].detach().requires_grad_(True)
    return t, x, y


# ============================================================
# PDE RESIDUAL AND LOSS
# ============================================================

def pde_residual(model, t, x, y, create_graph=True):
    u = model(t, x, y)

    u_t = grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_tt = grad(
        u_t, t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    u_x = grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_y = grad(
        u, y,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    Hux = H(x, y) * u_x
    Huy = H(x, y) * u_y

    dHux_dx = grad(
        Hux, x,
        grad_outputs=torch.ones_like(Hux),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    dHuy_dy = grad(
        Huy, y,
        grad_outputs=torch.ones_like(Huy),
        create_graph=create_graph,
        retain_graph=True,
    )[0]

    return u_tt - dHux_dx - dHuy_dy - f(t, x, y)


def loss_residual(model, t, x, y):
    residual = pde_residual(model, t, x, y, create_graph=True)
    return torch.mean(residual**2)


# ============================================================
# QUADRATURE AND ENERGY
# ============================================================

def trapz2d_weights(n, device, dtype):
    w = torch.ones(n, device=device, dtype=dtype)
    w[0] = 0.5
    w[-1] = 0.5
    return torch.outer(w, w)


def integrate_2d(values, weights, n):
    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)
    return torch.sum(weights * values) * dx * dy


def pinn_energy(model, t_scalar, n, device, dtype):
    x1d = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    y1d = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x1d, y1d, indexing="ij")

    t = torch.full((n * n, 1), float(t_scalar), device=device, dtype=dtype)
    x = X.reshape(-1, 1).detach().requires_grad_(True)
    y = Y.reshape(-1, 1).detach().requires_grad_(True)
    t = t.detach().requires_grad_(True)

    u = model(t, x, y)
    ones = torch.ones_like(u)
    u_t = grad(u, t, grad_outputs=ones, create_graph=False, retain_graph=True)[0]
    u_x = grad(u, x, grad_outputs=ones, create_graph=False, retain_graph=True)[0]
    u_y = grad(u, y, grad_outputs=ones, create_graph=False)[0]

    integrand = 0.5 * (u_t**2 + H(x, y) * (u_x**2 + u_y**2))
    integrand = integrand.reshape(n, n)
    weights = trapz2d_weights(n, device, dtype)
    return float(integrate_2d(integrand, weights, n).detach().cpu())


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(config, seed, verbose=True):
    dtype = get_dtype(config)
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    gc.collect()
    tracemalloc.start()

    base_net = MLP(
        hidden_layers=config.hidden_layers,
        width=config.width,
        T=T_final,
    ).to(device=DEVICE, dtype=dtype)
    model = PINNTrial(base_net).to(device=DEVICE, dtype=dtype)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # -----------------------------
    # Adam stage
    # -----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step,
        gamma=config.lr_gamma,
    )

    adam_losses = []
    adam_start = timeit.default_timer()

    for epoch in range(1, config.epochs_adam + 1):
        # New Sobol set at every epoch, but reproducible for each seed.
        t_pde, x_pde, y_pde = sample_collocation_sobol(
            config.N_pde_adam,
            dtype=dtype,
            device=DEVICE,
            seed=seed + epoch,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = loss_residual(model, t_pde, x_pde, y_pde)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        adam_losses.append(float(loss.detach().cpu()))

        if verbose and (epoch % 500 == 0 or epoch == config.epochs_adam):
            print(
                f"[{config.name} | seed={seed} | Adam] "
                f"epoch {epoch:5d} | loss={adam_losses[-1]:.6e} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    adam_time = timeit.default_timer() - adam_start

    # -----------------------------
    # L-BFGS stage
    # -----------------------------
    t_fix, x_fix, y_fix = sample_collocation_sobol(
        config.N_pde_lbfgs,
        dtype=dtype,
        device=DEVICE,
        seed=seed + 100000,
    )

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    lbfgs_losses = []

    def closure():
        optimizer_lbfgs.zero_grad(set_to_none=True)
        current_loss = loss_residual(model, t_fix, x_fix, y_fix)
        current_loss.backward()
        lbfgs_losses.append(float(current_loss.detach().cpu()))
        return current_loss

    lbfgs_start = timeit.default_timer()
    optimizer_lbfgs.step(closure)
    lbfgs_time = timeit.default_timer() - lbfgs_start

    # Recompute the actual final training loss at the final parameters.
    final_training_mse = float(
        loss_residual(model, t_fix, x_fix, y_fix).detach().cpu()
    )

    current_memory, peak_python_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_cuda_memory = 0.0
    if DEVICE.type == "cuda":
        peak_cuda_memory = torch.cuda.max_memory_allocated() / 1024**2

    training_info = {
        "parameter_count": parameter_count,
        "adam_losses": adam_losses,
        "lbfgs_losses": lbfgs_losses,
        "final_training_mse": final_training_mse,
        "adam_time_s": adam_time,
        "lbfgs_time_s": lbfgs_time,
        "training_time_s": adam_time + lbfgs_time,
        "peak_python_memory_mb": peak_python_memory / 1024**2,
        "peak_cuda_memory_mb": peak_cuda_memory,
    }
    return model, training_info


# ============================================================
# ERROR AND DIAGNOSTIC CALCULATIONS
# ============================================================

def evaluate_model(model, config, seed, full_energy_curve=True):
    dtype = get_dtype(config)
    n = config.n_final

    # -----------------------------
    # Final-time solution errors
    # -----------------------------
    eval_start = timeit.default_timer()

    x1d = torch.linspace(0.0, 1.0, n, device=DEVICE, dtype=dtype)
    y1d = torch.linspace(0.0, 1.0, n, device=DEVICE, dtype=dtype)
    X, Y = torch.meshgrid(x1d, y1d, indexing="ij")

    t_flat = torch.full((n * n, 1), T_final, device=DEVICE, dtype=dtype)
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    inference_start = timeit.default_timer()
    with torch.no_grad():
        u_pred = model(t_flat, x_flat, y_flat).reshape(n, n)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    inference_time = timeit.default_timer() - inference_start

    u_true = exact_u(t_flat, x_flat, y_flat).reshape(n, n)
    error = u_pred - u_true

    weights = trapz2d_weights(n, DEVICE, dtype)
    L2_error = torch.sqrt(integrate_2d(error**2, weights, n))
    L2_true = torch.sqrt(integrate_2d(u_true**2, weights, n))
    relative_L2 = L2_error / torch.clamp(L2_true, min=torch.finfo(dtype).eps)
    Linf_error = torch.max(torch.abs(error))

    # Relative H1 error at T.
    t_grad = t_flat.detach().clone()
    x_grad = x_flat.detach().clone().requires_grad_(True)
    y_grad = y_flat.detach().clone().requires_grad_(True)
    u_grad = model(t_grad, x_grad, y_grad)
    ux_pred = grad(
        u_grad, x_grad,
        grad_outputs=torch.ones_like(u_grad),
        create_graph=False,
        retain_graph=True,
    )[0].reshape(n, n)
    uy_pred = grad(
        u_grad, y_grad,
        grad_outputs=torch.ones_like(u_grad),
        create_graph=False,
    )[0].reshape(n, n)

    ux_true = exact_ux(t_flat, x_flat, y_flat).reshape(n, n)
    uy_true = exact_uy(t_flat, x_flat, y_flat).reshape(n, n)

    H1_error_sq = integrate_2d(
        error**2 + (ux_pred - ux_true) ** 2 + (uy_pred - uy_true) ** 2,
        weights,
        n,
    )
    H1_true_sq = integrate_2d(
        u_true**2 + ux_true**2 + uy_true**2,
        weights,
        n,
    )
    relative_H1 = torch.sqrt(H1_error_sq / torch.clamp(H1_true_sq, min=torch.finfo(dtype).eps))

    # -----------------------------
    # Independent space-time test
    # -----------------------------
    t_test, x_test, y_test = sample_collocation_sobol(
        config.N_test,
        dtype=dtype,
        device=DEVICE,
        seed=seed + 200000,
    )

    # Solution error does not need derivatives.
    with torch.no_grad():
        u_test_pred = model(t_test, x_test, y_test)
        u_test_true = exact_u(t_test, x_test, y_test)
        test_difference = u_test_pred - u_test_true
        space_time_relative_L2 = torch.sqrt(torch.mean(test_difference**2)) / torch.clamp(
            torch.sqrt(torch.mean(u_test_true**2)),
            min=torch.finfo(dtype).eps,
        )

    # Residual RMSE uses automatic differentiation on independent points.
    residual_test = pde_residual(
        model,
        t_test,
        x_test,
        y_test,
        create_graph=False,
    )
    independent_residual_rmse = torch.sqrt(torch.mean(residual_test**2))

    # -----------------------------
    # Energy errors
    # -----------------------------
    if full_energy_curve:
        times = np.linspace(0.0, T_final, config.n_energy_times)
    else:
        # The sensitivity table reports only the energy error at T.
        times = np.array([T_final])

    E_pinn = np.array([
        pinn_energy(model, float(t), config.n_energy, DEVICE, dtype)
        for t in times
    ])
    E_exact = exact_energy_np(times)
    relative_energy_errors = np.abs(E_pinn - E_exact) / np.maximum(np.abs(E_exact), 1.0e-14)

    relative_energy_error_T = float(relative_energy_errors[-1])
    maximum_relative_energy_error = (
        float(np.max(relative_energy_errors)) if full_energy_curve else np.nan
    )

    evaluation_time = timeit.default_timer() - eval_start

    metrics = {
        "L2_error_T": float(L2_error.detach().cpu()),
        "relative_L2_error_T": float(relative_L2.detach().cpu()),
        "Linf_error_T": float(Linf_error.detach().cpu()),
        "relative_H1_error_T": float(relative_H1.detach().cpu()),
        "space_time_relative_L2_error": float(space_time_relative_L2.detach().cpu()),
        "independent_residual_RMSE": float(independent_residual_rmse.detach().cpu()),
        "relative_energy_error_T": relative_energy_error_T,
        "maximum_relative_energy_error": maximum_relative_energy_error,
        "inference_time_s": inference_time,
        "evaluation_time_s": evaluation_time,
        "energy_times": times,
        "energy_pinn": E_pinn,
        "energy_exact": E_exact,
        "X": X.detach().cpu().numpy(),
        "Y": Y.detach().cpu().numpy(),
        "u_pred_T": u_pred.detach().cpu().numpy(),
        "u_true_T": u_true.detach().cpu().numpy(),
        "absolute_error_T": torch.abs(error).detach().cpu().numpy(),
    }
    return metrics


# ============================================================
# ONE COMPLETE RUN
# ============================================================

def run_experiment(config, seed, verbose=True, full_energy_curve=True):
    total_start = timeit.default_timer()
    model, training = train_model(config, seed, verbose=verbose)
    metrics = evaluate_model(model, config, seed, full_energy_curve=full_energy_curve)

    result = {
        "configuration": config.name,
        "seed": seed,
        "hidden_layers": config.hidden_layers,
        "width": config.width,
        "dtype": config.dtype_name,
        "Adam_points": config.N_pde_adam,
        "learning_rate": config.learning_rate,
        "LBFGS_points": config.N_pde_lbfgs,
        "parameter_count": training["parameter_count"],
        "final_training_residual_MSE": training["final_training_mse"],
        "relative_L2_error_T": metrics["relative_L2_error_T"],
        "Linf_error_T": metrics["Linf_error_T"],
        "relative_H1_error_T": metrics["relative_H1_error_T"],
        "space_time_relative_L2_error": metrics["space_time_relative_L2_error"],
        "independent_residual_RMSE": metrics["independent_residual_RMSE"],
        "relative_energy_error_T": metrics["relative_energy_error_T"],
        "maximum_relative_energy_error": metrics["maximum_relative_energy_error"],
        "Adam_training_time_s": training["adam_time_s"],
        "LBFGS_training_time_s": training["lbfgs_time_s"],
        "total_training_time_s": training["training_time_s"],
        "inference_time_s": metrics["inference_time_s"],
        "peak_python_memory_MB": training["peak_python_memory_mb"],
        "peak_CUDA_memory_MB": training["peak_cuda_memory_mb"],
        "total_run_time_s": timeit.default_timer() - total_start,
    }
    return model, training, metrics, result


# ============================================================
# ORIGINAL-STYLE PLOTS FOR ONE BASELINE RUN
# ============================================================

def moving_average(values, window=25):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def make_baseline_plots(training, metrics):
    X = metrics["X"]
    Y = metrics["Y"]
    U_true = metrics["u_true_T"]
    U_pred = metrics["u_pred_T"]
    Err = metrics["absolute_error_T"]

    # Exact solution, PINN prediction, absolute error.
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, Y, U_true, cmap="viridis")
    ax1.set_title(r"Exact $u(x,y,T)$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X, Y, U_pred, cmap="viridis")
    ax2.set_title(r"PINN $u_\theta(x,y,T)$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X, Y, Err, cmap="inferno")
    ax3.set_title(r"Error $|u_\theta-u|$")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("absolute error")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PINN_solution_comparison.png"), dpi=300)
    plt.show()

    # PINN energy against exact energy.
    plt.figure(figsize=(8, 5))
    plt.plot(
        metrics["energy_times"],
        metrics["energy_pinn"],
        "b-",
        label=r"PINN energy $E_\theta(t)$",
    )
    plt.plot(
        metrics["energy_times"],
        metrics["energy_exact"],
        "ro",
        fillstyle="none",
        markersize=4,
        label=r"Exact energy $E_{\mathrm{exact}}(t)$",
    )
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title("Energy: PINN versus exact solution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PINN_energy_comparison.png"), dpi=300)
    plt.show()

    # Adam and L-BFGS losses in separate panels.
    adam_losses = np.asarray(training["adam_losses"], dtype=float)
    lbfgs_losses = np.asarray(training["lbfgs_losses"], dtype=float)

    adam_window = 25
    lbfgs_window = 5
    adam_plot = moving_average(adam_losses, adam_window)
    lbfgs_plot = moving_average(lbfgs_losses, lbfgs_window)

    adam_x = np.arange(adam_window, adam_window + len(adam_plot))
    lbfgs_x = np.arange(1, len(lbfgs_plot) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(adam_x, adam_plot, linewidth=2)
    axes[0].set_xlabel("Adam epoch")
    axes[0].set_ylabel("Residual MSE loss")
    axes[0].set_title("Adam training")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(lbfgs_x, lbfgs_plot, linewidth=2)
    axes[1].set_xlabel("L-BFGS closure evaluation")
    axes[1].set_ylabel("Residual MSE loss")
    axes[1].set_title("L-BFGS refinement")
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "PINN_loss_Adam_LBFGS.png"), dpi=300)
    plt.show()


# ============================================================
# CSV AND SUMMARY UTILITIES
# ============================================================

RESULT_FIELDS = [
    "configuration",
    "seed",
    "hidden_layers",
    "width",
    "dtype",
    "Adam_points",
    "learning_rate",
    "LBFGS_points",
    "parameter_count",
    "final_training_residual_MSE",
    "relative_L2_error_T",
    "Linf_error_T",
    "relative_H1_error_T",
    "space_time_relative_L2_error",
    "independent_residual_RMSE",
    "relative_energy_error_T",
    "maximum_relative_energy_error",
    "Adam_training_time_s",
    "LBFGS_training_time_s",
    "total_training_time_s",
    "inference_time_s",
    "peak_python_memory_MB",
    "peak_CUDA_memory_MB",
    "total_run_time_s",
]


def write_results_csv(filename, rows):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def summarize_results(rows, group_name):
    summary_metrics = [
        "final_training_residual_MSE",
        "relative_L2_error_T",
        "Linf_error_T",
        "relative_H1_error_T",
        "space_time_relative_L2_error",
        "independent_residual_RMSE",
        "relative_energy_error_T",
        "maximum_relative_energy_error",
        "Adam_training_time_s",
        "LBFGS_training_time_s",
        "total_training_time_s",
        "inference_time_s",
        "peak_python_memory_MB",
        "peak_CUDA_memory_MB",
    ]

    summary = {"configuration": group_name, "number_of_runs": len(rows)}
    for metric in summary_metrics:
        mean, std = mean_std([row[metric] for row in rows])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
    return summary


def write_summary_csv(filename, summaries):
    if not summaries:
        return
    path = os.path.join(OUTPUT_DIR, filename)
    fieldnames = list(summaries[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved: {path}")


def print_baseline_table_values(summary):
    print("\nBaseline PINN: mean +/- standard deviation")
    labels = [
        ("Final training residual MSE", "final_training_residual_MSE"),
        ("Relative L2 error at T", "relative_L2_error_T"),
        ("Linf error at T", "Linf_error_T"),
        ("Relative H1 error at T", "relative_H1_error_T"),
        ("Space-time relative L2 error", "space_time_relative_L2_error"),
        ("Independent residual RMSE", "independent_residual_RMSE"),
        ("Relative energy error at T", "relative_energy_error_T"),
        ("Maximum relative energy error", "maximum_relative_energy_error"),
        ("Adam training time", "Adam_training_time_s"),
        ("L-BFGS training time", "LBFGS_training_time_s"),
        ("Inference time", "inference_time_s"),
        ("Peak Python memory", "peak_python_memory_MB"),
        ("Peak CUDA memory", "peak_CUDA_memory_MB"),
    ]
    for label, key in labels:
        mean = summary[f"{key}_mean"]
        std = summary[f"{key}_std"]
        print(f"{label:38s}: {mean:.6e} +/- {std:.6e}")


# ============================================================
# HYPERPARAMETER CONFIGURATIONS
# ============================================================

def sensitivity_configurations(baseline):
    """
    Exactly the configurations used in the proposed manuscript table.
    Only one baseline parameter is changed at a time.
    """
    return [
        replace(baseline, name="Baseline"),
        replace(baseline, name="Architecture 1", hidden_layers=3, width=32),
        replace(baseline, name="Architecture 2", hidden_layers=5, width=80),
        replace(baseline, name="Adam sampling 1", N_pde_adam=1000),
        replace(baseline, name="Adam sampling 2", N_pde_adam=6000),
        replace(baseline, name="Learning rate 1", learning_rate=1.0e-3),
        replace(baseline, name="Learning rate 2", learning_rate=5.0e-3),
        replace(baseline, name="L-BFGS sampling 1", N_pde_lbfgs=5000),
        replace(baseline, name="L-BFGS sampling 2", N_pde_lbfgs=20000),
        replace(baseline, name="Precision", dtype_name="float64"),
    ]


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    if FAST_TEST:
        print("FAST_TEST=True: these results are only a code check.")

    # --------------------------------------------------------
    # 1. Ordinary baseline run and the original figures
    # --------------------------------------------------------
    if RUN_BASELINE_PLOTS:
        print("\n=== Baseline plotting run ===")
        model, training, metrics, result = run_experiment(
            BASELINE,
            BASELINE_SEED,
            verbose=True,
        )

        print(f"Final training residual MSE = {result['final_training_residual_MSE']:.6e}")
        print(f"Relative L2 error at T       = {result['relative_L2_error_T']:.6e}")
        print(f"Linf error at T              = {result['Linf_error_T']:.6e}")
        print(f"Relative H1 error at T       = {result['relative_H1_error_T']:.6e}")
        print(f"Space-time relative L2       = {result['space_time_relative_L2_error']:.6e}")
        print(f"Independent residual RMSE    = {result['independent_residual_RMSE']:.6e}")
        print(f"Relative energy error at T   = {result['relative_energy_error_T']:.6e}")
        print(f"Maximum relative energy err. = {result['maximum_relative_energy_error']:.6e}")
        print(f"Adam time                    = {result['Adam_training_time_s']:.3f} s")
        print(f"L-BFGS time                  = {result['LBFGS_training_time_s']:.3f} s")
        print(f"Inference time               = {result['inference_time_s']:.6f} s")
        print(f"Peak Python memory           = {result['peak_python_memory_MB']:.3f} MB")
        if DEVICE.type == "cuda":
            print(f"Peak CUDA memory             = {result['peak_CUDA_memory_MB']:.3f} MB")

        write_results_csv("baseline_plot_run.csv", [result])
        make_baseline_plots(training, metrics)

        del model
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 2. Five baseline runs for the first manuscript table
    # --------------------------------------------------------
    if RUN_BASELINE_STATISTICS:
        print("\n=== Five-seed baseline statistics ===")
        baseline_rows = []

        for seed in STATISTICAL_SEEDS:
            print(f"\nBaseline seed {seed}")
            model, training, metrics, result = run_experiment(
                BASELINE,
                seed,
                verbose=False,
            )
            baseline_rows.append(result)

            del model
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        baseline_summary = summarize_results(baseline_rows, "Baseline")
        write_results_csv("baseline_five_seeds_raw.csv", baseline_rows)
        write_summary_csv("baseline_five_seeds_summary.csv", [baseline_summary])
        print_baseline_table_values(baseline_summary)

    # --------------------------------------------------------
    # 3. Hyperparameter sensitivity study for the second table
    # --------------------------------------------------------
    if RUN_SENSITIVITY_STUDY:
        print("\n=== Hyperparameter sensitivity study ===")
        all_rows = []
        all_summaries = []

        for config in sensitivity_configurations(BASELINE):
            print(f"\n--- {config.name} ---")
            configuration_rows = []

            for seed in STATISTICAL_SEEDS:
                print(f"{config.name}, seed {seed}")
                model, training, metrics, result = run_experiment(
                    config,
                    seed,
                    verbose=False,
                    full_energy_curve=False,
                )
                configuration_rows.append(result)
                all_rows.append(result)

                del model
                gc.collect()
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            all_summaries.append(
                summarize_results(configuration_rows, config.name)
            )

        write_results_csv("sensitivity_raw.csv", all_rows)
        write_summary_csv("sensitivity_summary.csv", all_summaries)

        print("\nSensitivity table: mean +/- standard deviation")
        for summary in all_summaries:
            name = summary["configuration"]
            print(
                f"{name:20s} | "
                f"rel L2={summary['relative_L2_error_T_mean']:.3e} +/- "
                f"{summary['relative_L2_error_T_std']:.3e} | "
                f"Linf={summary['Linf_error_T_mean']:.3e} +/- "
                f"{summary['Linf_error_T_std']:.3e} | "
                f"rel H1={summary['relative_H1_error_T_mean']:.3e} +/- "
                f"{summary['relative_H1_error_T_std']:.3e} | "
                f"res={summary['independent_residual_RMSE_mean']:.3e} +/- "
                f"{summary['independent_residual_RMSE_std']:.3e} | "
                f"energy={summary['relative_energy_error_T_mean']:.3e} +/- "
                f"{summary['relative_energy_error_T_std']:.3e} | "
                f"time={summary['total_training_time_s_mean']:.2f} +/- "
                f"{summary['total_training_time_s_std']:.2f} s"
            )
