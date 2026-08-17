"""
Fourth-order staggered FDM for the common heterogeneous benchmark

    u_tt - div(h(x,y) grad u) = 0  in (0,T) x (0,1)^2,
    u = 0 on the boundary,
    u(0,x,y) = boundary-compatible localized pulse,
    u_t(0,x,y) = 0.

The script produces:
  * a fine FDM reference solution on a common evaluation grid;
  * three FDM scalability levels (L1, L2, L3);
  * final-time L2, Linfinity and H1 errors relative to the FDM reference;
  * gauge waveform and arrival-time errors;
  * energy drift, timing and peak process-memory statistics;
  * CSV/NPZ files and manuscript-ready figures.

Run this script before the FEM and PINN scripts because they load the file
"fdm_complex_results/fdm_reference.npz".
"""

from __future__ import annotations

import csv
import os
import threading
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import psutil
except ImportError:  # pragma: no cover - fallback for minimal installations
    psutil = None


# =====================================================================
# USER OPTIONS
# =====================================================================

FAST_TEST = False
RUN_REFERENCE = True
RUN_SCALABILITY = True
MAKE_PLOTS = True

OUTPUT_DIR = "fdm_complex_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_FINAL = 0.75
CFL_NUMBER = 0.20
COMMON_GRID_N = 201
SNAPSHOT_TIMES = np.array([0.00, 0.25, 0.50, 0.75])
GAUGE_THRESHOLD = 1.0e-2
TIMING_REPEATS = 5

LEVELS = {
    "L1": 40,
    "L2": 80,
    "L3": 160,
}
REFERENCE_N = 320
REPRESENTATIVE_LEVEL = "L2"

if FAST_TEST:
    LEVELS = {"L1": 12, "L2": 16, "L3": 20}
    REFERENCE_N = 24
    COMMON_GRID_N = 41
    SNAPSHOT_TIMES = np.array([0.00, 0.375, 0.75])
    TIMING_REPEATS = 1


# =====================================================================
# COMMON BENCHMARK DATA
# =====================================================================

H_D = 1.0
H_S = 0.35
X_S = 0.72
W_S = 0.04
DELTA_H_M = 0.18
X_M, Y_M = 0.56, 0.62
A_M, B_M = 0.09, 0.07

AMPLITUDE = 1.0
X0, Y0 = 0.24, 0.48
A0, B0 = 0.055, 0.080

GAUGE_NAMES = np.array(["G1", "G2", "G3", "G4"])
GAUGES = np.array(
    [
        [0.38, 0.48],
        [0.56, 0.62],
        [0.69, 0.48],
        [0.84, 0.48],
    ],
    dtype=float,
)


def coefficient_h(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Smooth shelf plus localized seamount coefficient."""
    shelf = 0.5 * (H_D - H_S) * (1.0 - np.tanh((x - X_S) / W_S))
    seamount = DELTA_H_M * np.exp(
        -((x - X_M) / A_M) ** 2 - ((y - Y_M) / B_M) ** 2
    )
    return H_S + shelf + seamount


def initial_displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boundary-compatible asymmetric localized pulse."""
    denominator = X0 * (1.0 - X0) * Y0 * (1.0 - Y0)
    boundary_factor = x * (1.0 - x) * y * (1.0 - y) / denominator
    gaussian = np.exp(-((x - X0) / A0) ** 2 - ((y - Y0) / B0) ** 2)
    return AMPLITUDE * boundary_factor * gaussian


# =====================================================================
# MEMORY MONITOR
# =====================================================================

class PeakMemoryMonitor:
    """Sample peak process RSS; fall back to Python-traced memory."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline = 0
        self._peak = 0
        self._use_psutil = psutil is not None

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            rss = process.memory_info().rss
            self._peak = max(self._peak, rss)
            time.sleep(self.interval)

    def start(self) -> None:
        if self._use_psutil:
            process = psutil.Process(os.getpid())
            self._baseline = process.memory_info().rss
            self._peak = self._baseline
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        else:
            tracemalloc.start()

    def stop(self) -> float:
        if self._use_psutil:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()
            # Report peak total process RSS, not only the increment.
            return self._peak / 1024**2
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024**2


# =====================================================================
# FOURTH-ORDER STAGGERED OPERATOR
# =====================================================================

C_OS = np.array([-11 / 12, 17 / 24, 3 / 8, -5 / 24, 1 / 24])


def build_grids(Nx: int, Ny: int) -> Tuple[np.ndarray, ...]:
    if Nx < 6 or Ny < 6:
        raise ValueError("Nx and Ny must be at least 6.")
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)
    Xc, Yc = np.meshgrid(x, y, indexing="ij")

    xf = (np.arange(Nx) + 0.5) * dx
    yf = (np.arange(Ny) + 0.5) * dy
    Xfx, Yfx = np.meshgrid(xf, y, indexing="ij")
    Xfy, Yfy = np.meshgrid(x, yf, indexing="ij")
    return dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy


def grad_x_faces(u: np.ndarray, dx: float) -> np.ndarray:
    Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
    gx = np.zeros((Nx, Ny + 1), dtype=u.dtype)
    gx[0, :] = sum(C_OS[k] * u[k, :] for k in range(5)) / dx
    for i in range(1, Nx - 1):
        gx[i, :] = (
            u[i - 1, :] - 27.0 * u[i, :] + 27.0 * u[i + 1, :] - u[i + 2, :]
        ) / (24.0 * dx)
    gx[Nx - 1, :] = (
        11 / 12 * u[Nx, :]
        - 17 / 24 * u[Nx - 1, :]
        - 3 / 8 * u[Nx - 2, :]
        + 5 / 24 * u[Nx - 3, :]
        - 1 / 24 * u[Nx - 4, :]
    ) / dx
    return gx


def grad_y_faces(u: np.ndarray, dy: float) -> np.ndarray:
    Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
    gy = np.zeros((Nx + 1, Ny), dtype=u.dtype)
    gy[:, 0] = sum(C_OS[k] * u[:, k] for k in range(5)) / dy
    for j in range(1, Ny - 1):
        gy[:, j] = (
            u[:, j - 1] - 27.0 * u[:, j] + 27.0 * u[:, j + 1] - u[:, j + 2]
        ) / (24.0 * dy)
    gy[:, Ny - 1] = (
        11 / 12 * u[:, Ny]
        - 17 / 24 * u[:, Ny - 1]
        - 3 / 8 * u[:, Ny - 2]
        + 5 / 24 * u[:, Ny - 3]
        - 1 / 24 * u[:, Ny - 4]
    ) / dy
    return gy


def div_x_from_faces(Fx: np.ndarray, dx: float) -> np.ndarray:
    Nx, Ny = Fx.shape[0], Fx.shape[1] - 1
    divergence = np.zeros((Nx + 1, Ny + 1), dtype=Fx.dtype)
    divergence[1, :] = (
        -11 / 12 * Fx[0, :]
        + 17 / 24 * Fx[1, :]
        + 3 / 8 * Fx[2, :]
        - 5 / 24 * Fx[3, :]
        + 1 / 24 * Fx[4, :]
    ) / dx
    for i in range(2, Nx - 1):
        divergence[i, :] = (
            Fx[i - 2, :] - 27.0 * Fx[i - 1, :] + 27.0 * Fx[i, :] - Fx[i + 1, :]
        ) / (24.0 * dx)
    divergence[Nx - 1, :] = (
        11 / 12 * Fx[Nx - 1, :]
        - 17 / 24 * Fx[Nx - 2, :]
        - 3 / 8 * Fx[Nx - 3, :]
        + 5 / 24 * Fx[Nx - 4, :]
        - 1 / 24 * Fx[Nx - 5, :]
    ) / dx
    return divergence


def div_y_from_faces(Fy: np.ndarray, dy: float) -> np.ndarray:
    Nx, Ny = Fy.shape[0] - 1, Fy.shape[1]
    divergence = np.zeros((Nx + 1, Ny + 1), dtype=Fy.dtype)
    divergence[:, 1] = (
        -11 / 12 * Fy[:, 0]
        + 17 / 24 * Fy[:, 1]
        + 3 / 8 * Fy[:, 2]
        - 5 / 24 * Fy[:, 3]
        + 1 / 24 * Fy[:, 4]
    ) / dy
    for j in range(2, Ny - 1):
        divergence[:, j] = (
            Fy[:, j - 2] - 27.0 * Fy[:, j - 1] + 27.0 * Fy[:, j] - Fy[:, j + 1]
        ) / (24.0 * dy)
    divergence[:, Ny - 1] = (
        11 / 12 * Fy[:, Ny - 1]
        - 17 / 24 * Fy[:, Ny - 2]
        - 3 / 8 * Fy[:, Ny - 3]
        + 5 / 24 * Fy[:, Ny - 4]
        - 1 / 24 * Fy[:, Ny - 5]
    ) / dy
    return divergence


def L4_MAC(
    u: np.ndarray,
    dx: float,
    dy: float,
    Xfx: np.ndarray,
    Yfx: np.ndarray,
    Xfy: np.ndarray,
    Yfy: np.ndarray,
) -> np.ndarray:
    hfx = coefficient_h(Xfx, Yfx)
    hfy = coefficient_h(Xfy, Yfy)
    Fx = hfx * grad_x_faces(u, dx)
    Fy = hfy * grad_y_faces(u, dy)
    return div_x_from_faces(Fx, dx) + div_y_from_faces(Fy, dy)


def enforce_zero_boundary(u: np.ndarray) -> None:
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0


# =====================================================================
# POST-PROCESSING HELPERS
# =====================================================================


def bilinear_value(u: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
    Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
    px = min(max(x / dx, 0.0), Nx)
    py = min(max(y / dy, 0.0), Ny)
    i = min(int(np.floor(px)), Nx - 1)
    j = min(int(np.floor(py)), Ny - 1)
    tx = px - i
    ty = py - j
    return float(
        (1 - tx) * (1 - ty) * u[i, j]
        + tx * (1 - ty) * u[i + 1, j]
        + (1 - tx) * ty * u[i, j + 1]
        + tx * ty * u[i + 1, j + 1]
    )


def sample_gauges(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.array([bilinear_value(u, gx, gy, dx, dy) for gx, gy in GAUGES])


def energy_from_velocity(
    u: np.ndarray,
    velocity: np.ndarray,
    dx: float,
    dy: float,
    Xfx: np.ndarray,
    Yfx: np.ndarray,
    Xfy: np.ndarray,
    Yfy: np.ndarray,
) -> float:
    gx = grad_x_faces(u, dx)
    gy = grad_y_faces(u, dy)
    kinetic = 0.5 * np.sum(velocity**2) * dx * dy
    potential_x = 0.5 * np.sum(coefficient_h(Xfx, Yfx) * gx**2) * dx * dy
    potential_y = 0.5 * np.sum(coefficient_h(Xfy, Yfy) * gy**2) * dx * dy
    return float(kinetic + potential_x + potential_y)


def interpolate_field(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    common_x: np.ndarray,
    common_y: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (x, y), field, method="linear", bounds_error=False, fill_value=None
    )
    Xq, Yq = np.meshgrid(common_x, common_y, indexing="ij")
    points = np.column_stack([Xq.ravel(), Yq.ravel()])
    return interpolator(points).reshape(Xq.shape)


def trapz2d(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(np.trapz(values, y, axis=1), x, axis=0))


def field_metrics(
    field: np.ndarray,
    reference: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    difference = field - reference
    reference_norm = np.sqrt(max(trapz2d(reference**2, x, y), 1.0e-30))
    rel_l2 = np.sqrt(trapz2d(difference**2, x, y)) / reference_norm
    linf = float(np.max(np.abs(difference)))

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    field_x, field_y = np.gradient(field, dx, dy, edge_order=2)
    ref_x, ref_y = np.gradient(reference, dx, dy, edge_order=2)
    h1_num = trapz2d(
        difference**2 + (field_x - ref_x) ** 2 + (field_y - ref_y) ** 2,
        x,
        y,
    )
    h1_den = trapz2d(reference**2 + ref_x**2 + ref_y**2, x, y)
    rel_h1 = np.sqrt(h1_num / max(h1_den, 1.0e-30))
    return {"relative_L2": rel_l2, "Linf": linf, "relative_H1": rel_h1}


def arrival_time(times: np.ndarray, signal: np.ndarray, threshold: float) -> float:
    magnitude = np.abs(signal)
    indices = np.where(magnitude >= threshold)[0]
    if len(indices) == 0:
        return np.nan
    index = int(indices[0])
    if index == 0:
        return float(times[0])
    y0_value = magnitude[index - 1]
    y1_value = magnitude[index]
    if y1_value <= y0_value:
        return float(times[index])
    fraction = (threshold - y0_value) / (y1_value - y0_value)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def gauge_metrics(
    times: np.ndarray,
    gauge_values: np.ndarray,
    ref_times: np.ndarray,
    ref_values: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    interpolated = np.column_stack(
        [np.interp(ref_times, times, gauge_values[:, k]) for k in range(len(GAUGES))]
    )
    waveform_errors = np.zeros(len(GAUGES))
    arrival_errors = np.zeros(len(GAUGES))
    arrivals = np.zeros(len(GAUGES))
    ref_arrivals = np.zeros(len(GAUGES))

    for k in range(len(GAUGES)):
        numerator = np.trapz((interpolated[:, k] - ref_values[:, k]) ** 2, ref_times)
        denominator = np.trapz(ref_values[:, k] ** 2, ref_times)
        waveform_errors[k] = np.sqrt(numerator / max(denominator, 1.0e-30))
        arrivals[k] = arrival_time(ref_times, interpolated[:, k], GAUGE_THRESHOLD)
        ref_arrivals[k] = arrival_time(ref_times, ref_values[:, k], GAUGE_THRESHOLD)
        arrival_errors[k] = abs(arrivals[k] - ref_arrivals[k])

    return {
        "waveform_errors": waveform_errors,
        "mean_waveform_error": float(np.mean(waveform_errors)),
        "arrivals": arrivals,
        "reference_arrivals": ref_arrivals,
        "arrival_errors": arrival_errors,
        "maximum_arrival_error": float(np.nanmax(arrival_errors)),
        "interpolated_gauges": interpolated,
    }


# =====================================================================
# SOLVER
# =====================================================================

@dataclass
class SolverResult:
    N: int
    dt: float
    Nt: int
    x: np.ndarray
    y: np.ndarray
    final_field: np.ndarray
    times: np.ndarray
    gauge_values: np.ndarray
    energy_times: np.ndarray
    energies: np.ndarray
    snapshots: Dict[float, np.ndarray]
    solve_time_s: float
    peak_cpu_memory_mb: float


def solve_fdm(
    N: int,
    *,
    track_energy: bool = True,
    snapshot_times: Optional[Iterable[float]] = None,
) -> SolverResult:
    monitor = PeakMemoryMonitor()
    monitor.start()
    wall_start = time.perf_counter()

    dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy = build_grids(N, N)
    h_max = float(np.max(coefficient_h(Xc, Yc)))
    dt_target = CFL_NUMBER / (
        np.sqrt(h_max) * np.sqrt(dx ** (-2) + dy ** (-2))
    )
    Nt = int(np.ceil(T_FINAL / dt_target))
    dt = T_FINAL / Nt

    u0 = initial_displacement(Xc, Yc)
    enforce_zero_boundary(u0)
    velocity0 = np.zeros_like(u0)
    acceleration0 = L4_MAC(u0, dx, dy, Xfx, Yfx, Xfy, Yfy)

    u_prev = u0.copy()
    u_curr = u0 + dt * velocity0 + 0.5 * dt**2 * acceleration0
    enforce_zero_boundary(u_curr)

    times = np.linspace(0.0, T_FINAL, Nt + 1)
    gauge_values = np.zeros((Nt + 1, len(GAUGES)))
    gauge_values[0] = sample_gauges(u_prev, dx, dy)
    gauge_values[1] = sample_gauges(u_curr, dx, dy)

    energy_times = [0.0]
    energies = [energy_from_velocity(u_prev, velocity0, dx, dy, Xfx, Yfx, Xfy, Yfy)]

    if snapshot_times is None:
        snapshot_times = SNAPSHOT_TIMES
    requested_snapshots = sorted(float(t) for t in snapshot_times)
    snapshots: Dict[float, np.ndarray] = {}
    for target in requested_snapshots:
        if abs(target) < 0.5 * dt:
            snapshots[target] = u_prev.copy()

    u_before_prev: Optional[np.ndarray] = None
    for n in range(1, Nt):
        current_time = n * dt
        u_next = 2.0 * u_curr - u_prev + dt**2 * L4_MAC(
            u_curr, dx, dy, Xfx, Yfx, Xfy, Yfy
        )
        enforce_zero_boundary(u_next)

        gauge_values[n + 1] = sample_gauges(u_next, dx, dy)

        if track_energy:
            velocity = (u_next - u_prev) / (2.0 * dt)
            energy_times.append(current_time)
            energies.append(
                energy_from_velocity(u_curr, velocity, dx, dy, Xfx, Yfx, Xfy, Yfy)
            )

        next_time = (n + 1) * dt
        for target in requested_snapshots:
            if target not in snapshots and abs(next_time - target) <= 0.5 * dt:
                snapshots[target] = u_next.copy()

        u_before_prev = u_prev
        u_prev, u_curr = u_curr, u_next

    if track_energy:
        if u_before_prev is None:
            velocity_final = (u_curr - u_prev) / dt
        else:
            velocity_final = (3.0 * u_curr - 4.0 * u_prev + u_before_prev) / (2.0 * dt)
        energy_times.append(T_FINAL)
        energies.append(
            energy_from_velocity(u_curr, velocity_final, dx, dy, Xfx, Yfx, Xfy, Yfy)
        )

    for target in requested_snapshots:
        if target not in snapshots and abs(target - T_FINAL) <= 0.5 * dt:
            snapshots[target] = u_curr.copy()

    solve_time = time.perf_counter() - wall_start
    peak_memory = monitor.stop()

    return SolverResult(
        N=N,
        dt=dt,
        Nt=Nt,
        x=x,
        y=y,
        final_field=u_curr,
        times=times,
        gauge_values=gauge_values,
        energy_times=np.asarray(energy_times),
        energies=np.asarray(energies),
        snapshots=snapshots,
        solve_time_s=solve_time,
        peak_cpu_memory_mb=peak_memory,
    )


# =====================================================================
# OUTPUT AND PLOTS
# =====================================================================


def save_reference(result: SolverResult) -> str:
    common_x = np.linspace(0.0, 1.0, COMMON_GRID_N)
    common_y = np.linspace(0.0, 1.0, COMMON_GRID_N)
    final_common = interpolate_field(result.x, result.y, result.final_field, common_x, common_y)
    dx = common_x[1] - common_x[0]
    dy = common_y[1] - common_y[0]
    ux_common, uy_common = np.gradient(final_common, dx, dy, edge_order=2)

    snapshot_array = np.stack(
        [
            interpolate_field(result.x, result.y, result.snapshots[float(t)], common_x, common_y)
            for t in SNAPSHOT_TIMES
        ],
        axis=0,
    )

    path = os.path.join(OUTPUT_DIR, "fdm_reference.npz")
    np.savez_compressed(
        path,
        method="FDM",
        N=result.N,
        dt=result.dt,
        Nt=result.Nt,
        T=T_FINAL,
        common_x=common_x,
        common_y=common_y,
        final_field=final_common,
        final_ux=ux_common,
        final_uy=uy_common,
        gauge_names=GAUGE_NAMES,
        gauge_locations=GAUGES,
        gauge_times=result.times,
        gauge_values=result.gauge_values,
        energy_times=result.energy_times,
        energies=result.energies,
        snapshot_times=SNAPSHOT_TIMES,
        snapshots=snapshot_array,
    )
    print(f"Saved FDM reference: {path}")
    return path


def save_representative(result: SolverResult, metrics: Dict[str, float]) -> None:
    path = os.path.join(OUTPUT_DIR, "fdm_representative.npz")
    common_x = np.linspace(0.0, 1.0, COMMON_GRID_N)
    common_y = np.linspace(0.0, 1.0, COMMON_GRID_N)
    final_common = interpolate_field(result.x, result.y, result.final_field, common_x, common_y)
    np.savez_compressed(
        path,
        method="FDM",
        level=REPRESENTATIVE_LEVEL,
        N=result.N,
        dt=result.dt,
        Nt=result.Nt,
        common_x=common_x,
        common_y=common_y,
        final_field=final_common,
        gauge_times=result.times,
        gauge_values=result.gauge_values,
        energy_times=result.energy_times,
        energies=result.energies,
        relative_L2=metrics["relative_L2"],
        Linf=metrics["Linf"],
        relative_H1=metrics["relative_H1"],
    )
    print(f"Saved representative FDM data: {path}")


def make_setup_plot() -> None:
    x = np.linspace(0.0, 1.0, 301)
    y = np.linspace(0.0, 1.0, 301)
    X, Y = np.meshgrid(x, y, indexing="ij")
    H = coefficient_h(X, Y)
    U0 = initial_displacement(X, Y)

    fig = plt.figure(figsize=(14, 5.5))

    # Heterogeneous coefficient h(x,y) as a 3D surface.
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        X, Y, H, cmap="viridis", rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )

    # Plot gauge positions directly on the coefficient surface.
    gauge_z = coefficient_h(GAUGES[:, 0], GAUGES[:, 1])
    ax1.scatter(GAUGES[:, 0], GAUGES[:, 1], gauge_z, marker="o", s=40)
    for name, (gx, gy), gz in zip(GAUGE_NAMES, GAUGES, gauge_z):
        ax1.text(gx, gy, gz + 0.02, name)

    ax1.set_title("Heterogeneous coefficient and gauges")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel(r"$h(x,y)$")
    ax1.view_init(elev=30, azim=-135)
    fig.colorbar(surf1, ax=ax1, shrink=0.65, pad=0.10, label=r"$h(x,y)$")

    # Initial displacement u_0(x,y) as a 3D surface.
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf2 = ax2.plot_surface(
        X, Y, U0, cmap="viridis", rstride=1, cstride=1,
        linewidth=0, antialiased=True
    )
    ax2.set_title("Initial displacement")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel(r"$u_0(x,y)$")
    ax2.view_init(elev=30, azim=-135)
    fig.colorbar(surf2, ax=ax2, shrink=0.65, pad=0.10, label=r"$u_0(x,y)$")

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "heterogeneous_setup.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_fdm_plots(result: SolverResult, reference: np.lib.npyio.NpzFile) -> None:
    common_x = reference["common_x"]
    common_y = reference["common_y"]
    X, Y = np.meshgrid(common_x, common_y, indexing="ij")

    # Representative snapshots as 3D surface plots.
    fig = plt.figure(figsize=(18, 4.8))
    for k, target in enumerate(SNAPSHOT_TIMES):
        ax = fig.add_subplot(1, len(SNAPSHOT_TIMES), k + 1, projection="3d")
        field = interpolate_field(
            result.x, result.y, result.snapshots[float(target)], common_x, common_y
        )
        surf = ax.plot_surface(X, Y, field, cmap="viridis", edgecolor='black', linewidth=0.1, rstride=2, cstride=2, alpha=0.95)
        ax.set_title(fr"$t={target:.2f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r"$u(t,x,y)$")
        ax.view_init(elev=30, azim=110)
        fig.colorbar(surf, ax=ax, shrink=0.58, pad=0.10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "fdm_wavefield_snapshots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Gauges against reference.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        ax.plot(reference["gauge_times"], reference["gauge_values"][:, k], label="FDM reference")
        ax.plot(result.times, result.gauge_values[:, k], "--", label=f"FDM {REPRESENTATIVE_LEVEL}")
        ax.set_title(GAUGE_NAMES[k])
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fdm_gauge_records.png"), dpi=300)
    plt.close(fig)

    # Energy drift.
    relative_energy = (result.energies - result.energies[0]) / result.energies[0]
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(result.energy_times, relative_energy)
    plt.xlabel("t")
    plt.ylabel(r"$(E_h(t)-E_h(0))/E_h(0)$")
    plt.title("FDM relative energy variation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fdm_energy_drift.png"), dpi=300)
    plt.close()


# =====================================================================
# MAIN DRIVER
# =====================================================================


def main() -> None:
    print("Common heterogeneous benchmark: FDM")
    print(f"Output directory: {OUTPUT_DIR}")
    if FAST_TEST:
        print("FAST_TEST=True: do not use these numbers in the manuscript.")

    reference_path = os.path.join(OUTPUT_DIR, "fdm_reference.npz")
    if RUN_REFERENCE or not os.path.exists(reference_path):
        print(f"\nComputing FDM reference with N={REFERENCE_N} ...")
        reference_result = solve_fdm(REFERENCE_N, track_energy=True)
        reference_path = save_reference(reference_result)
        print(
            f"Reference: dt={reference_result.dt:.6e}, Nt={reference_result.Nt}, "
            f"time={reference_result.solve_time_s:.3f} s, "
            f"peak RSS={reference_result.peak_cpu_memory_mb:.2f} MB"
        )

    reference = np.load(reference_path, allow_pickle=False)
    common_x = reference["common_x"]
    common_y = reference["common_y"]
    ref_field = reference["final_field"]
    ref_times = reference["gauge_times"]
    ref_gauges = reference["gauge_values"]

    if MAKE_PLOTS:
        make_setup_plot()

    if not RUN_SCALABILITY:
        return

    # Small warm-up not included in the timing statistics.
    warmup_N = max(8, min(LEVELS.values()) // 2)
    print(f"\nWarm-up run with N={warmup_N} ...")
    _ = solve_fdm(warmup_N, track_energy=False, snapshot_times=[])

    csv_rows = []
    representative_result: Optional[SolverResult] = None
    representative_metrics: Optional[Dict[str, float]] = None

    for level, N in LEVELS.items():
        print(f"\nFDM {level}: N={N}")
        measured_results = []
        for repeat in range(TIMING_REPEATS):
            result = solve_fdm(
                N,
                track_energy=True,
                snapshot_times=SNAPSHOT_TIMES if repeat == 0 else [],
            )
            measured_results.append(result)
            print(
                f"  repeat {repeat + 1}/{TIMING_REPEATS}: "
                f"{result.solve_time_s:.3f} s, {result.peak_cpu_memory_mb:.2f} MB"
            )

        primary = measured_results[0]
        common_field = interpolate_field(
            primary.x, primary.y, primary.final_field, common_x, common_y
        )
        metrics = field_metrics(common_field, ref_field, common_x, common_y)
        gauge_info = gauge_metrics(
            primary.times, primary.gauge_values, ref_times, ref_gauges
        )
        energy_drift = float(
            np.max(np.abs(primary.energies - primary.energies[0]))
            / max(abs(primary.energies[0]), 1.0e-30)
        )

        times = np.array([item.solve_time_s for item in measured_results])
        memories = np.array([item.peak_cpu_memory_mb for item in measured_results])
        row = {
            "method": "FDM",
            "level": level,
            "N": N,
            "unknowns": (N - 1) ** 2,
            "dt": primary.dt,
            "time_steps": primary.Nt,
            "relative_L2": metrics["relative_L2"],
            "Linf": metrics["Linf"],
            "relative_H1": metrics["relative_H1"],
            "mean_gauge_error": gauge_info["mean_waveform_error"],
            "maximum_arrival_time_error": gauge_info["maximum_arrival_error"],
            "energy_drift": energy_drift,
            "solve_time_mean_s": float(np.mean(times)),
            "solve_time_std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "peak_cpu_memory_mean_MB": float(np.mean(memories)),
            "peak_cpu_memory_std_MB": float(np.std(memories, ddof=1)) if len(memories) > 1 else 0.0,
        }
        csv_rows.append(row)

        print(
            f"  rel L2={row['relative_L2']:.6e}, Linf={row['Linf']:.6e}, "
            f"rel H1={row['relative_H1']:.6e}"
        )
        print(
            f"  mean gauge error={row['mean_gauge_error']:.6e}, "
            f"max arrival error={row['maximum_arrival_time_error']:.6e}, "
            f"energy drift={row['energy_drift']:.6e}"
        )

        if level == REPRESENTATIVE_LEVEL:
            representative_result = primary
            representative_metrics = metrics
            save_representative(primary, metrics)

    csv_path = os.path.join(OUTPUT_DIR, "fdm_scalability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved FDM scalability table: {csv_path}")

    if MAKE_PLOTS and representative_result is not None and representative_metrics is not None:
        make_fdm_plots(representative_result, reference)

    print("\nFDM table values")
    for row in csv_rows:
        print(
            f"{row['level']}: N={row['N']}, unknowns={row['unknowns']}, "
            f"Nt={row['time_steps']}, relL2={row['relative_L2']:.6e}, "
            f"time={row['solve_time_mean_s']:.3f} +/- {row['solve_time_std_s']:.3f} s, "
            f"memory={row['peak_cpu_memory_mean_MB']:.2f} +/- "
            f"{row['peak_cpu_memory_std_MB']:.2f} MB"
        )


if __name__ == "__main__":
    main()

# """
# Fourth-order staggered FDM for the common heterogeneous benchmark
#
#     u_tt - div(h(x,y) grad u) = 0  in (0,T) x (0,1)^2,
#     u = 0 on the boundary,
#     u(0,x,y) = boundary-compatible localized pulse,
#     u_t(0,x,y) = 0.
#
# The script produces:
#   * a fine FDM reference solution on a common evaluation grid;
#   * three FDM scalability levels (L1, L2, L3);
#   * final-time L2, Linfinity and H1 errors relative to the FDM reference;
#   * gauge waveform and arrival-time errors;
#   * energy drift, timing and peak process-memory statistics;
#   * CSV/NPZ files and manuscript-ready figures.
#
# Run this script before the FEM and PINN scripts because they load the file
# "fdm_complex_results/fdm_reference.npz".
# """
#
# from __future__ import annotations
#
# import csv
# import os
# import threading
# import time
# import tracemalloc
# from dataclasses import dataclass
# from typing import Dict, Iterable, Optional, Tuple
#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import RegularGridInterpolator
#
# try:
#     import psutil
# except ImportError:  # pragma: no cover - fallback for minimal installations
#     psutil = None
#
#
# # =====================================================================
# # USER OPTIONS
# # =====================================================================
#
# FAST_TEST = False
# RUN_REFERENCE = True
# RUN_SCALABILITY = True
# MAKE_PLOTS = True
#
# OUTPUT_DIR = "fdm_complex_results"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# T_FINAL = 0.75
# CFL_NUMBER = 0.20
# COMMON_GRID_N = 201
# SNAPSHOT_TIMES = np.array([0.00, 0.25, 0.50, 0.75])
# GAUGE_THRESHOLD = 1.0e-2
# TIMING_REPEATS = 5
#
# LEVELS = {
#     "L1": 40,
#     "L2": 80,
#     "L3": 160,
# }
# REFERENCE_N = 320
# REPRESENTATIVE_LEVEL = "L2"
#
# if FAST_TEST:
#     LEVELS = {"L1": 12, "L2": 16, "L3": 20}
#     REFERENCE_N = 24
#     COMMON_GRID_N = 41
#     SNAPSHOT_TIMES = np.array([0.00, 0.375, 0.75])
#     TIMING_REPEATS = 1
#
#
# # =====================================================================
# # COMMON BENCHMARK DATA
# # =====================================================================
#
# H_D = 1.0
# H_S = 0.35
# X_S = 0.72
# W_S = 0.04
# DELTA_H_M = 0.18
# X_M, Y_M = 0.56, 0.62
# A_M, B_M = 0.09, 0.07
#
# AMPLITUDE = 1.0
# X0, Y0 = 0.24, 0.48
# A0, B0 = 0.055, 0.080
#
# GAUGE_NAMES = np.array(["G1", "G2", "G3", "G4"])
# GAUGES = np.array(
#     [
#         [0.38, 0.48],
#         [0.56, 0.62],
#         [0.69, 0.48],
#         [0.84, 0.48],
#     ],
#     dtype=float,
# )
#
#
# def coefficient_h(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """Smooth shelf plus localized seamount coefficient."""
#     shelf = 0.5 * (H_D - H_S) * (1.0 - np.tanh((x - X_S) / W_S))
#     seamount = DELTA_H_M * np.exp(
#         -((x - X_M) / A_M) ** 2 - ((y - Y_M) / B_M) ** 2
#     )
#     return H_S + shelf - seamount
#
#
# def initial_displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     """Boundary-compatible asymmetric localized pulse."""
#     denominator = X0 * (1.0 - X0) * Y0 * (1.0 - Y0)
#     boundary_factor = x * (1.0 - x) * y * (1.0 - y) / denominator
#     gaussian = np.exp(-((x - X0) / A0) ** 2 - ((y - Y0) / B0) ** 2)
#     return AMPLITUDE * boundary_factor * gaussian
#
#
# # =====================================================================
# # MEMORY MONITOR
# # =====================================================================
#
# class PeakMemoryMonitor:
#     """Sample peak process RSS; fall back to Python-traced memory."""
#
#     def __init__(self, interval: float = 0.01):
#         self.interval = interval
#         self._stop_event = threading.Event()
#         self._thread: Optional[threading.Thread] = None
#         self._baseline = 0
#         self._peak = 0
#         self._use_psutil = psutil is not None
#
#     def _sample(self) -> None:
#         process = psutil.Process(os.getpid())
#         while not self._stop_event.is_set():
#             rss = process.memory_info().rss
#             self._peak = max(self._peak, rss)
#             time.sleep(self.interval)
#
#     def start(self) -> None:
#         if self._use_psutil:
#             process = psutil.Process(os.getpid())
#             self._baseline = process.memory_info().rss
#             self._peak = self._baseline
#             self._thread = threading.Thread(target=self._sample, daemon=True)
#             self._thread.start()
#         else:
#             tracemalloc.start()
#
#     def stop(self) -> float:
#         if self._use_psutil:
#             self._stop_event.set()
#             if self._thread is not None:
#                 self._thread.join()
#             # Report peak total process RSS, not only the increment.
#             return self._peak / 1024**2
#         _, peak = tracemalloc.get_traced_memory()
#         tracemalloc.stop()
#         return peak / 1024**2
#
#
# # =====================================================================
# # FOURTH-ORDER STAGGERED OPERATOR
# # =====================================================================
#
# C_OS = np.array([-11 / 12, 17 / 24, 3 / 8, -5 / 24, 1 / 24])
#
#
# def build_grids(Nx: int, Ny: int) -> Tuple[np.ndarray, ...]:
#     if Nx < 6 or Ny < 6:
#         raise ValueError("Nx and Ny must be at least 6.")
#     dx = 1.0 / Nx
#     dy = 1.0 / Ny
#     x = np.linspace(0.0, 1.0, Nx + 1)
#     y = np.linspace(0.0, 1.0, Ny + 1)
#     Xc, Yc = np.meshgrid(x, y, indexing="ij")
#
#     xf = (np.arange(Nx) + 0.5) * dx
#     yf = (np.arange(Ny) + 0.5) * dy
#     Xfx, Yfx = np.meshgrid(xf, y, indexing="ij")
#     Xfy, Yfy = np.meshgrid(x, yf, indexing="ij")
#     return dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy
#
#
# def grad_x_faces(u: np.ndarray, dx: float) -> np.ndarray:
#     Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
#     gx = np.zeros((Nx, Ny + 1), dtype=u.dtype)
#     gx[0, :] = sum(C_OS[k] * u[k, :] for k in range(5)) / dx
#     for i in range(1, Nx - 1):
#         gx[i, :] = (
#             u[i - 1, :] - 27.0 * u[i, :] + 27.0 * u[i + 1, :] - u[i + 2, :]
#         ) / (24.0 * dx)
#     gx[Nx - 1, :] = (
#         11 / 12 * u[Nx, :]
#         - 17 / 24 * u[Nx - 1, :]
#         - 3 / 8 * u[Nx - 2, :]
#         + 5 / 24 * u[Nx - 3, :]
#         - 1 / 24 * u[Nx - 4, :]
#     ) / dx
#     return gx
#
#
# def grad_y_faces(u: np.ndarray, dy: float) -> np.ndarray:
#     Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
#     gy = np.zeros((Nx + 1, Ny), dtype=u.dtype)
#     gy[:, 0] = sum(C_OS[k] * u[:, k] for k in range(5)) / dy
#     for j in range(1, Ny - 1):
#         gy[:, j] = (
#             u[:, j - 1] - 27.0 * u[:, j] + 27.0 * u[:, j + 1] - u[:, j + 2]
#         ) / (24.0 * dy)
#     gy[:, Ny - 1] = (
#         11 / 12 * u[:, Ny]
#         - 17 / 24 * u[:, Ny - 1]
#         - 3 / 8 * u[:, Ny - 2]
#         + 5 / 24 * u[:, Ny - 3]
#         - 1 / 24 * u[:, Ny - 4]
#     ) / dy
#     return gy
#
#
# def div_x_from_faces(Fx: np.ndarray, dx: float) -> np.ndarray:
#     Nx, Ny = Fx.shape[0], Fx.shape[1] - 1
#     divergence = np.zeros((Nx + 1, Ny + 1), dtype=Fx.dtype)
#     divergence[1, :] = (
#         -11 / 12 * Fx[0, :]
#         + 17 / 24 * Fx[1, :]
#         + 3 / 8 * Fx[2, :]
#         - 5 / 24 * Fx[3, :]
#         + 1 / 24 * Fx[4, :]
#     ) / dx
#     for i in range(2, Nx - 1):
#         divergence[i, :] = (
#             Fx[i - 2, :] - 27.0 * Fx[i - 1, :] + 27.0 * Fx[i, :] - Fx[i + 1, :]
#         ) / (24.0 * dx)
#     divergence[Nx - 1, :] = (
#         11 / 12 * Fx[Nx - 1, :]
#         - 17 / 24 * Fx[Nx - 2, :]
#         - 3 / 8 * Fx[Nx - 3, :]
#         + 5 / 24 * Fx[Nx - 4, :]
#         - 1 / 24 * Fx[Nx - 5, :]
#     ) / dx
#     return divergence
#
#
# def div_y_from_faces(Fy: np.ndarray, dy: float) -> np.ndarray:
#     Nx, Ny = Fy.shape[0] - 1, Fy.shape[1]
#     divergence = np.zeros((Nx + 1, Ny + 1), dtype=Fy.dtype)
#     divergence[:, 1] = (
#         -11 / 12 * Fy[:, 0]
#         + 17 / 24 * Fy[:, 1]
#         + 3 / 8 * Fy[:, 2]
#         - 5 / 24 * Fy[:, 3]
#         + 1 / 24 * Fy[:, 4]
#     ) / dy
#     for j in range(2, Ny - 1):
#         divergence[:, j] = (
#             Fy[:, j - 2] - 27.0 * Fy[:, j - 1] + 27.0 * Fy[:, j] - Fy[:, j + 1]
#         ) / (24.0 * dy)
#     divergence[:, Ny - 1] = (
#         11 / 12 * Fy[:, Ny - 1]
#         - 17 / 24 * Fy[:, Ny - 2]
#         - 3 / 8 * Fy[:, Ny - 3]
#         + 5 / 24 * Fy[:, Ny - 4]
#         - 1 / 24 * Fy[:, Ny - 5]
#     ) / dy
#     return divergence
#
#
# def L4_MAC(
#     u: np.ndarray,
#     dx: float,
#     dy: float,
#     Xfx: np.ndarray,
#     Yfx: np.ndarray,
#     Xfy: np.ndarray,
#     Yfy: np.ndarray,
# ) -> np.ndarray:
#     hfx = coefficient_h(Xfx, Yfx)
#     hfy = coefficient_h(Xfy, Yfy)
#     Fx = hfx * grad_x_faces(u, dx)
#     Fy = hfy * grad_y_faces(u, dy)
#     return div_x_from_faces(Fx, dx) + div_y_from_faces(Fy, dy)
#
#
# def enforce_zero_boundary(u: np.ndarray) -> None:
#     u[0, :] = 0.0
#     u[-1, :] = 0.0
#     u[:, 0] = 0.0
#     u[:, -1] = 0.0
#
#
# # =====================================================================
# # POST-PROCESSING HELPERS
# # =====================================================================
#
#
# def bilinear_value(u: np.ndarray, x: float, y: float, dx: float, dy: float) -> float:
#     Nx, Ny = u.shape[0] - 1, u.shape[1] - 1
#     px = min(max(x / dx, 0.0), Nx)
#     py = min(max(y / dy, 0.0), Ny)
#     i = min(int(np.floor(px)), Nx - 1)
#     j = min(int(np.floor(py)), Ny - 1)
#     tx = px - i
#     ty = py - j
#     return float(
#         (1 - tx) * (1 - ty) * u[i, j]
#         + tx * (1 - ty) * u[i + 1, j]
#         + (1 - tx) * ty * u[i, j + 1]
#         + tx * ty * u[i + 1, j + 1]
#     )
#
#
# def sample_gauges(u: np.ndarray, dx: float, dy: float) -> np.ndarray:
#     return np.array([bilinear_value(u, gx, gy, dx, dy) for gx, gy in GAUGES])
#
#
# def energy_from_velocity(
#     u: np.ndarray,
#     velocity: np.ndarray,
#     dx: float,
#     dy: float,
#     Xfx: np.ndarray,
#     Yfx: np.ndarray,
#     Xfy: np.ndarray,
#     Yfy: np.ndarray,
# ) -> float:
#     gx = grad_x_faces(u, dx)
#     gy = grad_y_faces(u, dy)
#     kinetic = 0.5 * np.sum(velocity**2) * dx * dy
#     potential_x = 0.5 * np.sum(coefficient_h(Xfx, Yfx) * gx**2) * dx * dy
#     potential_y = 0.5 * np.sum(coefficient_h(Xfy, Yfy) * gy**2) * dx * dy
#     return float(kinetic + potential_x + potential_y)
#
#
# def interpolate_field(
#     x: np.ndarray,
#     y: np.ndarray,
#     field: np.ndarray,
#     common_x: np.ndarray,
#     common_y: np.ndarray,
# ) -> np.ndarray:
#     interpolator = RegularGridInterpolator(
#         (x, y), field, method="linear", bounds_error=False, fill_value=None
#     )
#     Xq, Yq = np.meshgrid(common_x, common_y, indexing="ij")
#     points = np.column_stack([Xq.ravel(), Yq.ravel()])
#     return interpolator(points).reshape(Xq.shape)
#
#
# def trapz2d(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
#     return float(np.trapz(np.trapz(values, y, axis=1), x, axis=0))
#
#
# def field_metrics(
#     field: np.ndarray,
#     reference: np.ndarray,
#     x: np.ndarray,
#     y: np.ndarray,
# ) -> Dict[str, float]:
#     difference = field - reference
#     reference_norm = np.sqrt(max(trapz2d(reference**2, x, y), 1.0e-30))
#     rel_l2 = np.sqrt(trapz2d(difference**2, x, y)) / reference_norm
#     linf = float(np.max(np.abs(difference)))
#
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     field_x, field_y = np.gradient(field, dx, dy, edge_order=2)
#     ref_x, ref_y = np.gradient(reference, dx, dy, edge_order=2)
#     h1_num = trapz2d(
#         difference**2 + (field_x - ref_x) ** 2 + (field_y - ref_y) ** 2,
#         x,
#         y,
#     )
#     h1_den = trapz2d(reference**2 + ref_x**2 + ref_y**2, x, y)
#     rel_h1 = np.sqrt(h1_num / max(h1_den, 1.0e-30))
#     return {"relative_L2": rel_l2, "Linf": linf, "relative_H1": rel_h1}
#
#
# def arrival_time(times: np.ndarray, signal: np.ndarray, threshold: float) -> float:
#     magnitude = np.abs(signal)
#     indices = np.where(magnitude >= threshold)[0]
#     if len(indices) == 0:
#         return np.nan
#     index = int(indices[0])
#     if index == 0:
#         return float(times[0])
#     y0_value = magnitude[index - 1]
#     y1_value = magnitude[index]
#     if y1_value <= y0_value:
#         return float(times[index])
#     fraction = (threshold - y0_value) / (y1_value - y0_value)
#     return float(times[index - 1] + fraction * (times[index] - times[index - 1]))
#
#
# def gauge_metrics(
#     times: np.ndarray,
#     gauge_values: np.ndarray,
#     ref_times: np.ndarray,
#     ref_values: np.ndarray,
# ) -> Dict[str, np.ndarray | float]:
#     interpolated = np.column_stack(
#         [np.interp(ref_times, times, gauge_values[:, k]) for k in range(len(GAUGES))]
#     )
#     waveform_errors = np.zeros(len(GAUGES))
#     arrival_errors = np.zeros(len(GAUGES))
#     arrivals = np.zeros(len(GAUGES))
#     ref_arrivals = np.zeros(len(GAUGES))
#
#     for k in range(len(GAUGES)):
#         numerator = np.trapz((interpolated[:, k] - ref_values[:, k]) ** 2, ref_times)
#         denominator = np.trapz(ref_values[:, k] ** 2, ref_times)
#         waveform_errors[k] = np.sqrt(numerator / max(denominator, 1.0e-30))
#         arrivals[k] = arrival_time(ref_times, interpolated[:, k], GAUGE_THRESHOLD)
#         ref_arrivals[k] = arrival_time(ref_times, ref_values[:, k], GAUGE_THRESHOLD)
#         arrival_errors[k] = abs(arrivals[k] - ref_arrivals[k])
#
#     return {
#         "waveform_errors": waveform_errors,
#         "mean_waveform_error": float(np.mean(waveform_errors)),
#         "arrivals": arrivals,
#         "reference_arrivals": ref_arrivals,
#         "arrival_errors": arrival_errors,
#         "maximum_arrival_error": float(np.nanmax(arrival_errors)),
#         "interpolated_gauges": interpolated,
#     }
#
#
# # =====================================================================
# # SOLVER
# # =====================================================================
#
# @dataclass
# class SolverResult:
#     N: int
#     dt: float
#     Nt: int
#     x: np.ndarray
#     y: np.ndarray
#     final_field: np.ndarray
#     times: np.ndarray
#     gauge_values: np.ndarray
#     energy_times: np.ndarray
#     energies: np.ndarray
#     snapshots: Dict[float, np.ndarray]
#     solve_time_s: float
#     peak_cpu_memory_mb: float
#
#
# def solve_fdm(
#     N: int,
#     *,
#     track_energy: bool = True,
#     snapshot_times: Optional[Iterable[float]] = None,
# ) -> SolverResult:
#     monitor = PeakMemoryMonitor()
#     monitor.start()
#     wall_start = time.perf_counter()
#
#     dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy = build_grids(N, N)
#     h_max = float(np.max(coefficient_h(Xc, Yc)))
#     dt_target = CFL_NUMBER / (
#         np.sqrt(h_max) * np.sqrt(dx ** (-2) + dy ** (-2))
#     )
#     Nt = int(np.ceil(T_FINAL / dt_target))
#     dt = T_FINAL / Nt
#
#     u0 = initial_displacement(Xc, Yc)
#     enforce_zero_boundary(u0)
#     velocity0 = np.zeros_like(u0)
#     acceleration0 = L4_MAC(u0, dx, dy, Xfx, Yfx, Xfy, Yfy)
#
#     u_prev = u0.copy()
#     u_curr = u0 + dt * velocity0 + 0.5 * dt**2 * acceleration0
#     enforce_zero_boundary(u_curr)
#
#     times = np.linspace(0.0, T_FINAL, Nt + 1)
#     gauge_values = np.zeros((Nt + 1, len(GAUGES)))
#     gauge_values[0] = sample_gauges(u_prev, dx, dy)
#     gauge_values[1] = sample_gauges(u_curr, dx, dy)
#
#     energy_times = [0.0]
#     energies = [energy_from_velocity(u_prev, velocity0, dx, dy, Xfx, Yfx, Xfy, Yfy)]
#
#     if snapshot_times is None:
#         snapshot_times = SNAPSHOT_TIMES
#     requested_snapshots = sorted(float(t) for t in snapshot_times)
#     snapshots: Dict[float, np.ndarray] = {}
#     for target in requested_snapshots:
#         if abs(target) < 0.5 * dt:
#             snapshots[target] = u_prev.copy()
#
#     u_before_prev: Optional[np.ndarray] = None
#     for n in range(1, Nt):
#         current_time = n * dt
#         u_next = 2.0 * u_curr - u_prev + dt**2 * L4_MAC(
#             u_curr, dx, dy, Xfx, Yfx, Xfy, Yfy
#         )
#         enforce_zero_boundary(u_next)
#
#         gauge_values[n + 1] = sample_gauges(u_next, dx, dy)
#
#         if track_energy:
#             velocity = (u_next - u_prev) / (2.0 * dt)
#             energy_times.append(current_time)
#             energies.append(
#                 energy_from_velocity(u_curr, velocity, dx, dy, Xfx, Yfx, Xfy, Yfy)
#             )
#
#         next_time = (n + 1) * dt
#         for target in requested_snapshots:
#             if target not in snapshots and abs(next_time - target) <= 0.5 * dt:
#                 snapshots[target] = u_next.copy()
#
#         u_before_prev = u_prev
#         u_prev, u_curr = u_curr, u_next
#
#     if track_energy:
#         if u_before_prev is None:
#             velocity_final = (u_curr - u_prev) / dt
#         else:
#             velocity_final = (3.0 * u_curr - 4.0 * u_prev + u_before_prev) / (2.0 * dt)
#         energy_times.append(T_FINAL)
#         energies.append(
#             energy_from_velocity(u_curr, velocity_final, dx, dy, Xfx, Yfx, Xfy, Yfy)
#         )
#
#     for target in requested_snapshots:
#         if target not in snapshots and abs(target - T_FINAL) <= 0.5 * dt:
#             snapshots[target] = u_curr.copy()
#
#     solve_time = time.perf_counter() - wall_start
#     peak_memory = monitor.stop()
#
#     return SolverResult(
#         N=N,
#         dt=dt,
#         Nt=Nt,
#         x=x,
#         y=y,
#         final_field=u_curr,
#         times=times,
#         gauge_values=gauge_values,
#         energy_times=np.asarray(energy_times),
#         energies=np.asarray(energies),
#         snapshots=snapshots,
#         solve_time_s=solve_time,
#         peak_cpu_memory_mb=peak_memory,
#     )
#
#
# # =====================================================================
# # OUTPUT AND PLOTS
# # =====================================================================
#
#
# def save_reference(result: SolverResult) -> str:
#     common_x = np.linspace(0.0, 1.0, COMMON_GRID_N)
#     common_y = np.linspace(0.0, 1.0, COMMON_GRID_N)
#     final_common = interpolate_field(result.x, result.y, result.final_field, common_x, common_y)
#     dx = common_x[1] - common_x[0]
#     dy = common_y[1] - common_y[0]
#     ux_common, uy_common = np.gradient(final_common, dx, dy, edge_order=2)
#
#     snapshot_array = np.stack(
#         [
#             interpolate_field(result.x, result.y, result.snapshots[float(t)], common_x, common_y)
#             for t in SNAPSHOT_TIMES
#         ],
#         axis=0,
#     )
#
#     path = os.path.join(OUTPUT_DIR, "fdm_reference.npz")
#     np.savez_compressed(
#         path,
#         method="FDM",
#         N=result.N,
#         dt=result.dt,
#         Nt=result.Nt,
#         T=T_FINAL,
#         common_x=common_x,
#         common_y=common_y,
#         final_field=final_common,
#         final_ux=ux_common,
#         final_uy=uy_common,
#         gauge_names=GAUGE_NAMES,
#         gauge_locations=GAUGES,
#         gauge_times=result.times,
#         gauge_values=result.gauge_values,
#         energy_times=result.energy_times,
#         energies=result.energies,
#         snapshot_times=SNAPSHOT_TIMES,
#         snapshots=snapshot_array,
#     )
#     print(f"Saved FDM reference: {path}")
#     return path
#
#
# def save_representative(result: SolverResult, metrics: Dict[str, float]) -> None:
#     path = os.path.join(OUTPUT_DIR, "fdm_representative.npz")
#     common_x = np.linspace(0.0, 1.0, COMMON_GRID_N)
#     common_y = np.linspace(0.0, 1.0, COMMON_GRID_N)
#     final_common = interpolate_field(result.x, result.y, result.final_field, common_x, common_y)
#     np.savez_compressed(
#         path,
#         method="FDM",
#         level=REPRESENTATIVE_LEVEL,
#         N=result.N,
#         dt=result.dt,
#         Nt=result.Nt,
#         common_x=common_x,
#         common_y=common_y,
#         final_field=final_common,
#         gauge_times=result.times,
#         gauge_values=result.gauge_values,
#         energy_times=result.energy_times,
#         energies=result.energies,
#         relative_L2=metrics["relative_L2"],
#         Linf=metrics["Linf"],
#         relative_H1=metrics["relative_H1"],
#     )
#     print(f"Saved representative FDM data: {path}")
#
#
# def make_setup_plot() -> None:
#     x = np.linspace(0.0, 1.0, 301)
#     y = np.linspace(0.0, 1.0, 301)
#     X, Y = np.meshgrid(x, y, indexing="ij")
#     H = coefficient_h(X, Y)
#     U0 = initial_displacement(X, Y)
#
#     fig, axes = plt.subplots(1, 2, figsize=(13, 5))
#     image0 = axes[0].contourf(X, Y, H, levels=40)
#     fig.colorbar(image0, ax=axes[0], label=r"$h(x,y)$")
#     axes[0].scatter(GAUGES[:, 0], GAUGES[:, 1], marker="o")
#     for name, (gx, gy) in zip(GAUGE_NAMES, GAUGES):
#         axes[0].text(gx + 0.012, gy + 0.012, name)
#     axes[0].set_title("Heterogeneous coefficient and gauges")
#     axes[0].set_xlabel("x")
#     axes[0].set_ylabel("y")
#     axes[0].set_aspect("equal")
#
#     image1 = axes[1].contourf(X, Y, U0, levels=40)
#     fig.colorbar(image1, ax=axes[1], label=r"$u_0(x,y)$")
#     axes[1].set_title("Initial displacement")
#     axes[1].set_xlabel("x")
#     axes[1].set_ylabel("y")
#     axes[1].set_aspect("equal")
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "heterogeneous_setup.png"), dpi=300)
#     plt.close(fig)
#
#
# def make_fdm_plots(result: SolverResult, reference: np.lib.npyio.NpzFile) -> None:
#     common_x = reference["common_x"]
#     common_y = reference["common_y"]
#     X, Y = np.meshgrid(common_x, common_y, indexing="ij")
#
#     # Representative snapshots.
#     fig, axes = plt.subplots(1, len(SNAPSHOT_TIMES), figsize=(17, 4))
#     for ax, target in zip(axes, SNAPSHOT_TIMES):
#         field = interpolate_field(
#             result.x, result.y, result.snapshots[float(target)], common_x, common_y
#         )
#         contour = ax.contourf(X, Y, field, levels=40)
#         fig.colorbar(contour, ax=ax, shrink=0.78)
#         ax.set_title(fr"$t={target:.2f}$")
#         ax.set_xlabel("x")
#         ax.set_ylabel("y")
#         ax.set_aspect("equal")
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "fdm_wavefield_snapshots.png"), dpi=300)
#     plt.close(fig)
#
#     # Gauges against reference.
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
#     for k, ax in enumerate(axes.ravel()):
#         ax.plot(reference["gauge_times"], reference["gauge_values"][:, k], label="FDM reference")
#         ax.plot(result.times, result.gauge_values[:, k], "--", label=f"FDM {REPRESENTATIVE_LEVEL}")
#         ax.set_title(GAUGE_NAMES[k])
#         ax.set_ylabel("u")
#         ax.grid(True, alpha=0.3)
#     axes[-1, 0].set_xlabel("t")
#     axes[-1, 1].set_xlabel("t")
#     axes[0, 0].legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "fdm_gauge_records.png"), dpi=300)
#     plt.close(fig)
#
#     # Energy drift.
#     relative_energy = (result.energies - result.energies[0]) / result.energies[0]
#     plt.figure(figsize=(7.5, 4.8))
#     plt.plot(result.energy_times, relative_energy)
#     plt.xlabel("t")
#     plt.ylabel(r"$(E_h(t)-E_h(0))/E_h(0)$")
#     plt.title("FDM relative energy variation")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "fdm_energy_drift.png"), dpi=300)
#     plt.close()
#
#
# # =====================================================================
# # MAIN DRIVER
# # =====================================================================
#
#
# def main() -> None:
#     print("Common heterogeneous benchmark: FDM")
#     print(f"Output directory: {OUTPUT_DIR}")
#     if FAST_TEST:
#         print("FAST_TEST=True: do not use these numbers in the manuscript.")
#
#     reference_path = os.path.join(OUTPUT_DIR, "fdm_reference.npz")
#     if RUN_REFERENCE or not os.path.exists(reference_path):
#         print(f"\nComputing FDM reference with N={REFERENCE_N} ...")
#         reference_result = solve_fdm(REFERENCE_N, track_energy=True)
#         reference_path = save_reference(reference_result)
#         print(
#             f"Reference: dt={reference_result.dt:.6e}, Nt={reference_result.Nt}, "
#             f"time={reference_result.solve_time_s:.3f} s, "
#             f"peak RSS={reference_result.peak_cpu_memory_mb:.2f} MB"
#         )
#
#     reference = np.load(reference_path, allow_pickle=False)
#     common_x = reference["common_x"]
#     common_y = reference["common_y"]
#     ref_field = reference["final_field"]
#     ref_times = reference["gauge_times"]
#     ref_gauges = reference["gauge_values"]
#
#     if MAKE_PLOTS:
#         make_setup_plot()
#
#     if not RUN_SCALABILITY:
#         return
#
#     # Small warm-up not included in the timing statistics.
#     warmup_N = max(8, min(LEVELS.values()) // 2)
#     print(f"\nWarm-up run with N={warmup_N} ...")
#     _ = solve_fdm(warmup_N, track_energy=False, snapshot_times=[])
#
#     csv_rows = []
#     representative_result: Optional[SolverResult] = None
#     representative_metrics: Optional[Dict[str, float]] = None
#
#     for level, N in LEVELS.items():
#         print(f"\nFDM {level}: N={N}")
#         measured_results = []
#         for repeat in range(TIMING_REPEATS):
#             result = solve_fdm(
#                 N,
#                 track_energy=True,
#                 snapshot_times=SNAPSHOT_TIMES if repeat == 0 else [],
#             )
#             measured_results.append(result)
#             print(
#                 f"  repeat {repeat + 1}/{TIMING_REPEATS}: "
#                 f"{result.solve_time_s:.3f} s, {result.peak_cpu_memory_mb:.2f} MB"
#             )
#
#         primary = measured_results[0]
#         common_field = interpolate_field(
#             primary.x, primary.y, primary.final_field, common_x, common_y
#         )
#         metrics = field_metrics(common_field, ref_field, common_x, common_y)
#         gauge_info = gauge_metrics(
#             primary.times, primary.gauge_values, ref_times, ref_gauges
#         )
#         energy_drift = float(
#             np.max(np.abs(primary.energies - primary.energies[0]))
#             / max(abs(primary.energies[0]), 1.0e-30)
#         )
#
#         times = np.array([item.solve_time_s for item in measured_results])
#         memories = np.array([item.peak_cpu_memory_mb for item in measured_results])
#         row = {
#             "method": "FDM",
#             "level": level,
#             "N": N,
#             "unknowns": (N - 1) ** 2,
#             "dt": primary.dt,
#             "time_steps": primary.Nt,
#             "relative_L2": metrics["relative_L2"],
#             "Linf": metrics["Linf"],
#             "relative_H1": metrics["relative_H1"],
#             "mean_gauge_error": gauge_info["mean_waveform_error"],
#             "maximum_arrival_time_error": gauge_info["maximum_arrival_error"],
#             "energy_drift": energy_drift,
#             "solve_time_mean_s": float(np.mean(times)),
#             "solve_time_std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
#             "peak_cpu_memory_mean_MB": float(np.mean(memories)),
#             "peak_cpu_memory_std_MB": float(np.std(memories, ddof=1)) if len(memories) > 1 else 0.0,
#         }
#         csv_rows.append(row)
#
#         print(
#             f"  rel L2={row['relative_L2']:.6e}, Linf={row['Linf']:.6e}, "
#             f"rel H1={row['relative_H1']:.6e}"
#         )
#         print(
#             f"  mean gauge error={row['mean_gauge_error']:.6e}, "
#             f"max arrival error={row['maximum_arrival_time_error']:.6e}, "
#             f"energy drift={row['energy_drift']:.6e}"
#         )
#
#         if level == REPRESENTATIVE_LEVEL:
#             representative_result = primary
#             representative_metrics = metrics
#             save_representative(primary, metrics)
#
#     csv_path = os.path.join(OUTPUT_DIR, "fdm_scalability.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as file:
#         writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(csv_rows)
#     print(f"\nSaved FDM scalability table: {csv_path}")
#
#     if MAKE_PLOTS and representative_result is not None and representative_metrics is not None:
#         make_fdm_plots(representative_result, reference)
#
#     print("\nFDM table values")
#     for row in csv_rows:
#         print(
#             f"{row['level']}: N={row['N']}, unknowns={row['unknowns']}, "
#             f"Nt={row['time_steps']}, relL2={row['relative_L2']:.6e}, "
#             f"time={row['solve_time_mean_s']:.3f} +/- {row['solve_time_std_s']:.3f} s, "
#             f"memory={row['peak_cpu_memory_mean_MB']:.2f} +/- "
#             f"{row['peak_cpu_memory_std_MB']:.2f} MB"
#         )
#
#
# if __name__ == "__main__":
#     main()
