# ============================================================
# Model problem:
# PINN for u_tt - div(h ∇u) = f on (t,x,y) in [0,T] × [0,1]^2
# IC and BC:
#   u(0,x,y) = u0(x,y),
#   u_t(0,x,y) = u1(x,y) (here u1 ≡ 0),
#   u|_{∂Ω} = 0.
# Trial:
#   u_theta(t,x,y) = u0(x,y) + t*u1(x,y)
#                    + t^2 * x(1-x) y(1-y) * N_theta( t', x', y' ).
# For this test problem, u1(x,y) ≡ 0.
# ============================================================

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import timeit
import tracemalloc

tracemalloc.start()  # start tracing
# -----------------------------
# Setup
# -----------------------------

start = timeit.default_timer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32
torch.set_default_dtype(dtype)
torch.manual_seed(1234)
np.random.seed(1234)

# Final time
T_final = 1.0

# -----------------------------
# Exact data and coefficients
# -----------------------------
def u0(x, y):
    """Initial displacement u(0,x,y)."""
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def u1(x, y):
    """Initial velocity u_t(0,x,y). For this test: identically zero."""
    return torch.zeros_like(x)

def exact_solution(t, x, y):
    return torch.cos(t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def h_coef(x, y):
    return x**2 + y**2

def f_func(t, x, y):
    # Right-hand side corresponding to u_tt - div(h ∇u) = f
    sxs = torch.sin(torch.pi * x); cxs = torch.cos(torch.pi * x)
    sys = torch.sin(torch.pi * y); cys = torch.cos(torch.pi * y)
    term1 = sxs * sys
    term2 = 2.0 * torch.pi * x * cxs * sys
    term3 = 2.0 * torch.pi * y * sxs * cys
    term4 = 2.0 * (torch.pi**2) * (x**2 + y**2) * sxs * sys
    return -torch.cos(t) * (term1 + term2 + term3 - term4)

# Exact energy: E_exact(t) = 1/8 sin^2 t + (π^2/6) cos^2 t
def exact_energy_np(t):
    t = np.asarray(t, dtype=float)
    return 0.125 * np.sin(t)**2 + (np.pi**2 / 6.0) * np.cos(t)**2

# -----------------------------
# MLP (with input normalization inside)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3, width=64, depth=4, act=nn.Tanh(), T=T_final):
        super().__init__()
        self.T = T
        layers = []
        dims = [in_dim] + [width] * (depth - 1) + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(m.bias)

    def forward(self, t, x, y):
        # Normalise inputs to [-1,1]^3
        t_hat = 2.0 * t / self.T - 1.0
        x_hat = 2.0 * x - 1.0
        y_hat = 2.0 * y - 1.0
        inp = torch.cat([t_hat, x_hat, y_hat], dim=1)
        return self.net(inp)

# -----------------------------
# Trial solution wrapper (hard constraints)
# -----------------------------
class PINNTrial(nn.Module):
    """
    Trial solution:
      u_theta(t,x,y) = u0(x,y) + t*u1(x,y)
                       + t^2 * x(1-x) y(1-y) * N_theta(t,x,y)
    For this problem, u1 ≡ 0, so we recover the original ansatz.
    """
    def __init__(self, base_net: nn.Module):
        super().__init__()
        self.net = base_net

    def forward(self, t, x, y):
        g = (t**2) * (x * (1.0 - x)) * (y * (1.0 - y))
        return u0(x, y) + t * u1(x, y) + g * self.net(t, x, y)

# -----------------------------
# Collocation sampling (Sobol)
# -----------------------------
def sample_collocation_sobol(N, T=T_final, device=device):
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
    pts = engine.draw(N).to(device=device, dtype=dtype)
    t = pts[:, 0:1] * T
    x = pts[:, 1:2]
    y = pts[:, 2:3]
    t.requires_grad_(True); x.requires_grad_(True); y.requires_grad_(True)
    return t, x, y

# -----------------------------
# Residual and loss
# -----------------------------
def pde_residual_theta(u_model: PINNTrial, t, x, y):
    """
    PDE residual:
      R_theta = u_tt - ∂_x(h u_x) - ∂_y(h u_y) - f
    with all derivatives computed by automatic differentiation.
    """
    u = u_model(t, x, y)

    # Time derivatives
    u_t  = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    # Spatial derivatives
    u_x  = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y  = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    hv = h_coef(x, y)
    hux = hv * u_x
    duy_dx = grad(hux, x, grad_outputs=torch.ones_like(hux), create_graph=True)[0]
    huy = hv * u_y
    duy_dy = grad(huy, y, grad_outputs=torch.ones_like(huy), create_graph=True)[0]

    R_theta = u_tt - (duy_dx + duy_dy) - f_func(t, x, y)
    return R_theta

def loss_residual(u_model: PINNTrial, t, x, y):
    R_theta = pde_residual_theta(u_model, t, x, y)
    return torch.mean(R_theta**2)

# -----------------------------
# Build model
# -----------------------------
width, depth = 64, 5
base_net = MLP(in_dim=3, width=width, depth=depth, act=nn.Tanh(), T=T_final).to(device)
model = PINNTrial(base_net).to(device)

# -----------------------------
# Training (two-phase: Adam -> L-BFGS)
# -----------------------------
epochs_adam   = 3000
N_pde_adam    = 3000
lr            = 2e-3
optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

adam_losses, adam_epochs = [], []
lbfgs_losses, lbfgs_calls = [], []

for ep in range(1, epochs_adam + 1):
    t_pde, x_pde, y_pde = sample_collocation_sobol(N_pde_adam, T=T_final, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_residual(model, t_pde, x_pde, y_pde)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    adam_losses.append(loss.item()); adam_epochs.append(ep)
    if ep % 500 == 0:
        print(f"[Adam] Epoch {ep:5d} | Loss = {loss.item():.6e} | LR = {scheduler.get_last_lr()[0]:.2e}")

# L-BFGS refinement
N_pde_lbfgs = 10000
t_fix, x_fix, y_fix = sample_collocation_sobol(N_pde_lbfgs, T=T_final, device=device)
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), max_iter=500, tolerance_grad=1e-9,
    tolerance_change=1e-12, line_search_fn='strong_wolfe'
)

def closure():
    optimizer_lbfgs.zero_grad(set_to_none=True)
    L = loss_residual(model, t_fix, x_fix, y_fix)
    L.backward()
    lbfgs_losses.append(L.item()); lbfgs_calls.append(len(lbfgs_losses))
    return L

print("[L-BFGS] Starting refinement ...")
final_loss = optimizer_lbfgs.step(closure)
print(f"[L-BFGS] Done. Final loss = {final_loss.item():.6e}")

# -----------------------------
# Evaluation grid utils
# -----------------------------
def grid_xy(n=101, device=device):
    x = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1,1)
    y = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1,1)
    X, Y = torch.meshgrid(x.squeeze(1), y.squeeze(1), indexing='ij')
    return X, Y

# 2D trapezoidal rule weights on [0,1]x[0,1]
def trapz2d_weights(n, device=None, dtype=None):
    w1 = torch.ones(n, dtype=dtype, device=device)
    w1[0] = 0.5; w1[-1] = 0.5
    return torch.outer(w1, w1)  # (n,n)

# Compute discrete energy on an n×n grid
def pinn_energy(model, t_scalar, n=129, device=device, dtype=dtype):
    # grid (make leaves with grads)
    x1d = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1,1)
    y1d = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1,1)
    X, Y = torch.meshgrid(x1d.squeeze(1), y1d.squeeze(1), indexing='ij')

    # create leaf tensors requiring grads
    t = torch.full((n, n, 1), float(t_scalar), device=device, dtype=dtype, requires_grad=True)
    x = X.unsqueeze(-1).detach().clone().requires_grad_(True)   # (n,n,1)
    y = Y.unsqueeze(-1).detach().clone().requires_grad_(True)   # (n,n,1)

    # forward
    u = model(t.reshape(-1,1), x.reshape(-1,1), y.reshape(-1,1)).reshape(n, n, 1)

    # first derivatives (no higher-order graph needed)
    ones = torch.ones_like(u)
    u_t = grad(u, t, grad_outputs=ones, create_graph=False, retain_graph=True)[0]
    u_x = grad(u, x, grad_outputs=ones, create_graph=False, retain_graph=True)[0]
    u_y = grad(u, y, grad_outputs=ones, create_graph=False, retain_graph=True)[0]

    hv = h_coef(x, y)  # (n,n,1)
    integrand = 0.5 * (u_t**2 + hv * (u_x**2 + u_y**2))  # (n,n,1)
    integrand2d = integrand.squeeze(-1)

    W  = trapz2d_weights(n, device=device, dtype=dtype)
    dx = 1.0/(n-1); dy = 1.0/(n-1)
    Eh = torch.sum(W * integrand2d) * dx * dy
    return float(Eh)

def pinn_energy_series(model, t_array, n=129):
    return np.array([pinn_energy(model, float(tt), n=n) for tt in np.asarray(t_array)])

# -----------------------------
# Evaluation (final time T_final)
# -----------------------------
with torch.no_grad():
    nvis = 101
    X, Y = grid_xy(nvis, device=device)
    tT = torch.full_like(X, fill_value=T_final)
    u_pred_T = model(tT.reshape(-1,1), X.reshape(-1,1), Y.reshape(-1,1)).reshape(nvis, nvis)
    u_true_T = exact_solution(tT, X, Y).reshape(nvis, nvis)
    errT = torch.abs(u_pred_T - u_true_T)
    max_err = errT.max().item()
    l2_err  = torch.sqrt(torch.mean((u_pred_T - u_true_T)**2)).item()

print(f"[Eval @ T={T_final}]  Max error = {max_err:.6e},  L2 error = {l2_err:.6e}")

# Energy at T and energy curve over time
E_pinn_T  = pinn_energy(model, T_final, n=129)
E_exact_T = exact_energy_np(T_final)
print(f"[Energy @ T={T_final}]  PINN = {E_pinn_T:.8e}  |  exact = {E_exact_T:.8e}  |  rel.err = {abs(E_pinn_T-E_exact_T)/max(1e-14,E_exact_T):.3e}")

ts = np.linspace(0.0, T_final, 41)
E_pinn_series = np.array([pinn_energy(model, float(tt), n=129) for tt in ts])
E_exact_series = exact_energy_np(ts)

# -----------------------------
# Visualization (final time)
# -----------------------------
u_PINN = u_pred_T.detach().cpu().numpy()
U_true = u_true_T.detach().cpu().numpy()
Err    = np.abs(u_PINN - U_true)
Xn, Yn = X.detach().cpu().numpy(), Y.detach().cpu().numpy()

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(Xn, Yn, U_true, cmap='viridis')
ax1.set_title(r'Exact $u(x,y,T)$'); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(Xn, Yn, u_PINN, cmap='viridis')
ax2.set_title(r'PINN $u_\theta(x,y,T)$'); ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(Xn, Yn, Err, cmap='inferno')
ax3.set_title(r'Error $|u_\theta-u|$'); ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('abs error')

plt.tight_layout()
plt.show()

# Energy plot (PINN vs exact)
plt.figure(figsize=(8,5))
plt.plot(ts, E_pinn_series, 'b-',  markersize=4, label='PINN energy $E_h(t)$')
plt.plot(ts, E_exact_series,  'ro', fillstyle='none', markersize=4, label='Exact energy $E_{\mathrm{exact}}(t)$')
plt.xlabel('t'); plt.ylabel('Energy'); plt.title('Energy (PINN vs Exact)')
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# -----------------------------
# Loss curves (Adam + L-BFGS)
# -----------------------------
def moving_average(a, k=25):
    if k <= 1:
        return np.asarray(a)
    a = np.asarray(a, dtype=float)
    if len(a) < k:
        return a
    cs = np.cumsum(np.insert(a, 0, 0.0))
    return (cs[k:] - cs[:-k]) / k

SMOOTH = True
K = 25

adam_x = np.array(adam_epochs)
adam_y = np.array(adam_losses)
lbfgs_x = adam_x[-1] + np.arange(1, len(lbfgs_losses) + 1)
lbfgs_y = np.array(lbfgs_losses)

if SMOOTH:
    adam_y_plot  = moving_average(adam_y, K);      adam_x_plot  = adam_x[K-1:]
    lbfgs_y_plot = moving_average(lbfgs_y, max(5, K//5))
    lbfgs_x_plot = lbfgs_x[max(5, K//5)-1:]
else:
    adam_x_plot, adam_y_plot = adam_x, adam_y
    lbfgs_x_plot, lbfgs_y_plot = lbfgs_x, lbfgs_y

plt.figure(figsize=(8,5))
plt.semilogy(adam_x_plot,  adam_y_plot,  label='Adam',  linewidth=2)
plt.semilogy(lbfgs_x_plot, lbfgs_y_plot, label='L-BFGS (closure evals)', linewidth=2)
plt.xlabel('Iteration'); plt.ylabel('Residual MSE Loss'); plt.title('Training Loss (Adam → L-BFGS)')
plt.grid(True, which='both', alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")

tracemalloc.stop()
