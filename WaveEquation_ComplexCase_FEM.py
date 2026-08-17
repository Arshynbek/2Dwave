"""
Mass-lumped P1 FEM with explicit leapfrog time stepping for the common
heterogeneous benchmark.

Run TsunamiEquation_ComplexCase_FDM.py first. This script loads
"fdm_complex_results/fdm_reference.npz" and produces:
  * an independently refined FEM reference solution;
  * FDM--FEM reference-agreement diagnostics;
  * three FEM scalability levels;
  * field, gauge, arrival-time and energy diagnostics;
  * timing and peak process-memory statistics;
  * CSV/NPZ files and figures for the manuscript.
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
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


# =====================================================================
# USER OPTIONS
# =====================================================================

FAST_TEST = False
RUN_FEM_REFERENCE = True
RUN_SCALABILITY = True
MAKE_PLOTS = True

OUTPUT_DIR = "fem_complex_results"
FDM_REFERENCE_FILE = os.path.join("fdm_complex_results", "fdm_reference.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

T_FINAL = 0.75
CFL_NUMBER = 0.15
SNAPSHOT_TIMES = np.array([0.00, 0.25, 0.50, 0.75])
GAUGE_THRESHOLD = 1.0e-2
TIMING_REPEATS = 5

LEVELS = {
    "L1": 40,
    "L2": 80,
    "L3": 160,
}
REFERENCE_N = 240
REPRESENTATIVE_LEVEL = "L2"

if FAST_TEST:
    LEVELS = {"L1": 8, "L2": 12, "L3": 16}
    REFERENCE_N = 20
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
    [[0.38, 0.48], [0.56, 0.62], [0.69, 0.48], [0.84, 0.48]],
    dtype=float,
)


def coefficient_h(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    shelf = 0.5 * (H_D - H_S) * (1.0 - np.tanh((x - X_S) / W_S))
    seamount = DELTA_H_M * np.exp(
        -((x - X_M) / A_M) ** 2 - ((y - Y_M) / B_M) ** 2
    )
    return H_S + shelf + seamount


def initial_displacement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    denominator = X0 * (1.0 - X0) * Y0 * (1.0 - Y0)
    boundary_factor = x * (1.0 - x) * y * (1.0 - y) / denominator
    gaussian = np.exp(-((x - X0) / A0) ** 2 - ((y - Y0) / B0) ** 2)
    return AMPLITUDE * boundary_factor * gaussian


# =====================================================================
# MEMORY MONITOR
# =====================================================================

class PeakMemoryMonitor:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0
        self._use_psutil = psutil is not None

    def _sample(self) -> None:
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            self._peak = max(self._peak, process.memory_info().rss)
            time.sleep(self.interval)

    def start(self) -> None:
        if self._use_psutil:
            process = psutil.Process(os.getpid())
            self._peak = process.memory_info().rss
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        else:
            tracemalloc.start()

    def stop(self) -> float:
        if self._use_psutil:
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join()
            return self._peak / 1024**2
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024**2


# =====================================================================
# MESH AND ASSEMBLY
# =====================================================================

QUAD_BARY = np.array(
    [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]],
    dtype=float,
)
QUAD_W = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)


def generate_mesh(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    hx, hy = 1.0 / nx, 1.0 / ny
    nodes = np.array(
        [(i * hx, j * hy) for j in range(ny + 1) for i in range(nx + 1)],
        dtype=float,
    )
    elems = np.empty((2 * nx * ny, 3), dtype=np.int64)
    counter = 0
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1)
            v3 = v2 + 1
            elems[counter] = [v0, v1, v3]
            elems[counter + 1] = [v0, v3, v2]
            counter += 2
    return nodes, elems


def tri_area_and_grads(X: np.ndarray) -> Tuple[float, np.ndarray]:
    (x1, y1), (x2, y2), (x3, y3) = X
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(detJ)
    grads = np.array(
        [
            [(y2 - y3) / detJ, (x3 - x2) / detJ],
            [(y3 - y1) / detJ, (x1 - x3) / detJ],
            [(y1 - y2) / detJ, (x2 - x1) / detJ],
        ]
    )
    return area, grads


def assemble_lumped_mass_and_stiffness(
    nodes: np.ndarray, elems: np.ndarray
) -> Tuple[np.ndarray, sp.csr_matrix]:
    number_of_nodes = len(nodes)
    mass_diag = np.zeros(number_of_nodes)
    entries_per_element = 9
    total_entries = entries_per_element * len(elems)
    rows = np.empty(total_entries, dtype=np.int64)
    cols = np.empty(total_entries, dtype=np.int64)
    values = np.empty(total_entries, dtype=float)

    position = 0
    for tri in elems:
        X = nodes[tri]
        area, grads = tri_area_and_grads(X)
        mass_diag[tri] += area / 3.0

        Kloc = np.zeros((3, 3))
        grad_product = grads @ grads.T
        for q in range(3):
            point = QUAD_BARY[q] @ X
            hq = coefficient_h(point[0], point[1])
            Kloc += QUAD_W[q] * area * hq * grad_product

        local_rows = np.repeat(tri, 3)
        local_cols = np.tile(tri, 3)
        rows[position : position + 9] = local_rows
        cols[position : position + 9] = local_cols
        values[position : position + 9] = Kloc.ravel()
        position += 9

    stiffness = sp.coo_matrix(
        (values, (rows, cols)), shape=(number_of_nodes, number_of_nodes)
    ).tocsr()
    return mass_diag, stiffness


def interior_boundary_sets(nodes: np.ndarray, tolerance: float = 1.0e-14):
    boundary = np.where(
        (np.abs(nodes[:, 0]) < tolerance)
        | (np.abs(nodes[:, 0] - 1.0) < tolerance)
        | (np.abs(nodes[:, 1]) < tolerance)
        | (np.abs(nodes[:, 1] - 1.0) < tolerance)
    )[0]
    interior = np.setdiff1d(np.arange(len(nodes)), boundary)
    return interior, boundary


def nodal_grid(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Convert node ordering (y-major) to an array indexed as [x,y]."""
    return values.reshape((ny + 1, nx + 1)).T


# =====================================================================
# POST-PROCESSING HELPERS
# =====================================================================


def bilinear_value(grid: np.ndarray, x: float, y: float, hx: float, hy: float) -> float:
    nx, ny = grid.shape[0] - 1, grid.shape[1] - 1
    px = min(max(x / hx, 0.0), nx)
    py = min(max(y / hy, 0.0), ny)
    i = min(int(np.floor(px)), nx - 1)
    j = min(int(np.floor(py)), ny - 1)
    tx, ty = px - i, py - j
    return float(
        (1 - tx) * (1 - ty) * grid[i, j]
        + tx * (1 - ty) * grid[i + 1, j]
        + (1 - tx) * ty * grid[i, j + 1]
        + tx * ty * grid[i + 1, j + 1]
    )


def sample_gauges(grid: np.ndarray, hx: float, hy: float) -> np.ndarray:
    return np.array([bilinear_value(grid, gx, gy, hx, hy) for gx, gy in GAUGES])


def interpolate_field(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    common_x: np.ndarray,
    common_y: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator((x, y), field, bounds_error=False, fill_value=None)
    Xq, Yq = np.meshgrid(common_x, common_y, indexing="ij")
    points = np.column_stack([Xq.ravel(), Yq.ravel()])
    return interpolator(points).reshape(Xq.shape)


def trapz2d(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(np.trapz(values, y, axis=1), x, axis=0))


def field_metrics(field: np.ndarray, reference: np.ndarray, x: np.ndarray, y: np.ndarray):
    difference = field - reference
    ref_norm = np.sqrt(max(trapz2d(reference**2, x, y), 1.0e-30))
    relative_l2 = np.sqrt(trapz2d(difference**2, x, y)) / ref_norm
    linf = float(np.max(np.abs(difference)))
    dx, dy = x[1] - x[0], y[1] - y[0]
    ux, uy = np.gradient(field, dx, dy, edge_order=2)
    ref_ux, ref_uy = np.gradient(reference, dx, dy, edge_order=2)
    numerator = trapz2d(difference**2 + (ux - ref_ux) ** 2 + (uy - ref_uy) ** 2, x, y)
    denominator = trapz2d(reference**2 + ref_ux**2 + ref_uy**2, x, y)
    relative_h1 = np.sqrt(numerator / max(denominator, 1.0e-30))
    return {"relative_L2": relative_l2, "Linf": linf, "relative_H1": relative_h1}


def arrival_time(times: np.ndarray, signal: np.ndarray, threshold: float) -> float:
    magnitude = np.abs(signal)
    indices = np.where(magnitude >= threshold)[0]
    if len(indices) == 0:
        return np.nan
    index = int(indices[0])
    if index == 0:
        return float(times[0])
    low, high = magnitude[index - 1], magnitude[index]
    if high <= low:
        return float(times[index])
    fraction = (threshold - low) / (high - low)
    return float(times[index - 1] + fraction * (times[index] - times[index - 1]))


def gauge_metrics(times, values, ref_times, ref_values):
    interpolated = np.column_stack(
        [np.interp(ref_times, times, values[:, k]) for k in range(len(GAUGES))]
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
        "mean_waveform_error": float(np.mean(waveform_errors)),
        "waveform_errors": waveform_errors,
        "maximum_arrival_error": float(np.nanmax(arrival_errors)),
        "arrival_errors": arrival_errors,
        "arrivals": arrivals,
        "reference_arrivals": ref_arrivals,
        "interpolated_gauges": interpolated,
    }


def fem_energy(U: np.ndarray, V: np.ndarray, mass_diag: np.ndarray, K: sp.csr_matrix) -> float:
    kinetic = 0.5 * float(np.dot(mass_diag * V, V))
    potential = 0.5 * float(U @ (K @ U))
    return kinetic + potential


# =====================================================================
# SOLVER
# =====================================================================

@dataclass
class SolverResult:
    N: int
    dt: float
    Nt: int
    nodes: np.ndarray
    elems: np.ndarray
    x: np.ndarray
    y: np.ndarray
    final_field: np.ndarray
    times: np.ndarray
    gauge_values: np.ndarray
    energy_times: np.ndarray
    energies: np.ndarray
    snapshots: Dict[float, np.ndarray]
    assembly_time_s: float
    solve_time_s: float
    total_time_s: float
    peak_cpu_memory_mb: float


def solve_fem(
    N: int,
    *,
    track_energy: bool = True,
    snapshot_times: Optional[Iterable[float]] = None,
) -> SolverResult:
    monitor = PeakMemoryMonitor()
    monitor.start()
    total_start = time.perf_counter()

    assembly_start = time.perf_counter()
    nodes, elems = generate_mesh(N, N)
    mass_diag, K = assemble_lumped_mass_and_stiffness(nodes, elems)
    interior, _ = interior_boundary_sets(nodes)
    assembly_time = time.perf_counter() - assembly_start

    h = 1.0 / N
    h_max = float(np.max(coefficient_h(nodes[:, 0], nodes[:, 1])))
    dt_target = CFL_NUMBER * h / np.sqrt(h_max)
    Nt = int(np.ceil(T_FINAL / dt_target))
    dt = T_FINAL / Nt

    solve_start = time.perf_counter()
    U0 = initial_displacement(nodes[:, 0], nodes[:, 1])
    V0 = np.zeros_like(U0)
    U0[np.setdiff1d(np.arange(len(nodes)), interior)] = 0.0

    mass_interior = mass_diag[interior]
    Kii = K[interior, :][:, interior].tocsr()
    acceleration0 = -(Kii @ U0[interior]) / mass_interior

    U_prev = U0.copy()
    U_curr = np.zeros_like(U0)
    U_curr[interior] = U0[interior] + dt * V0[interior] + 0.5 * dt**2 * acceleration0

    times = np.linspace(0.0, T_FINAL, Nt + 1)
    gauge_values = np.zeros((Nt + 1, len(GAUGES)))
    hx = hy = h
    gauge_values[0] = sample_gauges(nodal_grid(U_prev, N, N), hx, hy)
    gauge_values[1] = sample_gauges(nodal_grid(U_curr, N, N), hx, hy)

    energy_times = [0.0]
    energies = [fem_energy(U_prev, V0, mass_diag, K)]

    if snapshot_times is None:
        snapshot_times = SNAPSHOT_TIMES
    requested_snapshots = sorted(float(value) for value in snapshot_times)
    snapshots: Dict[float, np.ndarray] = {}
    for target in requested_snapshots:
        if abs(target) < 0.5 * dt:
            snapshots[target] = nodal_grid(U_prev, N, N).copy()

    U_before_prev: Optional[np.ndarray] = None
    for n in range(1, Nt):
        current_time = n * dt
        U_next = np.zeros_like(U_curr)
        U_next[interior] = (
            2.0 * U_curr[interior]
            - U_prev[interior]
            - dt**2 * (Kii @ U_curr[interior]) / mass_interior
        )

        gauge_values[n + 1] = sample_gauges(nodal_grid(U_next, N, N), hx, hy)

        if track_energy:
            velocity = (U_next - U_prev) / (2.0 * dt)
            energy_times.append(current_time)
            energies.append(fem_energy(U_curr, velocity, mass_diag, K))

        next_time = (n + 1) * dt
        for target in requested_snapshots:
            if target not in snapshots and abs(next_time - target) <= 0.5 * dt:
                snapshots[target] = nodal_grid(U_next, N, N).copy()

        U_before_prev = U_prev
        U_prev, U_curr = U_curr, U_next

    if track_energy:
        if U_before_prev is None:
            V_final = (U_curr - U_prev) / dt
        else:
            V_final = (3.0 * U_curr - 4.0 * U_prev + U_before_prev) / (2.0 * dt)
        energy_times.append(T_FINAL)
        energies.append(fem_energy(U_curr, V_final, mass_diag, K))

    for target in requested_snapshots:
        if target not in snapshots and abs(target - T_FINAL) <= 0.5 * dt:
            snapshots[target] = nodal_grid(U_curr, N, N).copy()

    solve_time = time.perf_counter() - solve_start
    total_time = time.perf_counter() - total_start
    peak_memory = monitor.stop()

    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    return SolverResult(
        N=N,
        dt=dt,
        Nt=Nt,
        nodes=nodes,
        elems=elems,
        x=x,
        y=y,
        final_field=nodal_grid(U_curr, N, N),
        times=times,
        gauge_values=gauge_values,
        energy_times=np.asarray(energy_times),
        energies=np.asarray(energies),
        snapshots=snapshots,
        assembly_time_s=assembly_time,
        solve_time_s=solve_time,
        total_time_s=total_time,
        peak_cpu_memory_mb=peak_memory,
    )


# =====================================================================
# OUTPUT AND PLOTS
# =====================================================================


def save_fem_reference(result: SolverResult, fdm_reference) -> str:
    common_x = fdm_reference["common_x"]
    common_y = fdm_reference["common_y"]
    final_common = interpolate_field(result.x, result.y, result.final_field, common_x, common_y)
    snapshot_array = np.stack(
        [
            interpolate_field(result.x, result.y, result.snapshots[float(t)], common_x, common_y)
            for t in SNAPSHOT_TIMES
        ],
        axis=0,
    )
    path = os.path.join(OUTPUT_DIR, "fem_reference.npz")
    np.savez_compressed(
        path,
        method="FEM",
        N=result.N,
        dt=result.dt,
        Nt=result.Nt,
        common_x=common_x,
        common_y=common_y,
        final_field=final_common,
        gauge_names=GAUGE_NAMES,
        gauge_locations=GAUGES,
        gauge_times=result.times,
        gauge_values=result.gauge_values,
        energy_times=result.energy_times,
        energies=result.energies,
        snapshot_times=SNAPSHOT_TIMES,
        snapshots=snapshot_array,
    )
    print(f"Saved FEM reference: {path}")
    return path


def save_representative(result: SolverResult, metrics: Dict[str, float]) -> None:
    path = os.path.join(OUTPUT_DIR, "fem_representative.npz")
    np.savez_compressed(
        path,
        method="FEM",
        level=REPRESENTATIVE_LEVEL,
        N=result.N,
        dt=result.dt,
        Nt=result.Nt,
        x=result.x,
        y=result.y,
        final_field=result.final_field,
        gauge_times=result.times,
        gauge_values=result.gauge_values,
        energy_times=result.energy_times,
        energies=result.energies,
        relative_L2=metrics["relative_L2"],
        Linf=metrics["Linf"],
        relative_H1=metrics["relative_H1"],
    )
    print(f"Saved representative FEM data: {path}")


def make_fem_plots(result: SolverResult, fdm_reference) -> None:
    common_x, common_y = fdm_reference["common_x"], fdm_reference["common_y"]
    X, Y = np.meshgrid(common_x, common_y, indexing="ij")

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
    plt.savefig(os.path.join(OUTPUT_DIR, "fem_wavefield_snapshots.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, ax in enumerate(axes.ravel()):
        ax.plot(fdm_reference["gauge_times"], fdm_reference["gauge_values"][:, k], label="FDM reference")
        ax.plot(result.times, result.gauge_values[:, k], "--", label=f"FEM {REPRESENTATIVE_LEVEL}")
        ax.set_title(GAUGE_NAMES[k])
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fem_gauge_records.png"), dpi=300)
    plt.close(fig)

    relative_energy = (result.energies - result.energies[0]) / result.energies[0]
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(result.energy_times, relative_energy)
    plt.xlabel("t")
    plt.ylabel(r"$(E_h(t)-E_h(0))/E_h(0)$")
    plt.title("FEM relative energy variation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fem_energy_drift.png"), dpi=300)
    plt.close()


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
# =====================================================================
# MAIN DRIVER
# =====================================================================


def main() -> None:
    print("Common heterogeneous benchmark: FEM")
    print(f"Output directory: {OUTPUT_DIR}")
    if FAST_TEST:
        print("FAST_TEST=True: do not use these numbers in the manuscript.")

    if not os.path.exists(FDM_REFERENCE_FILE):
        raise FileNotFoundError(
            f"FDM reference not found: {FDM_REFERENCE_FILE}. "
            "Run TsunamiEquation_ComplexCase_FDM.py first."
        )
    fdm_reference = np.load(FDM_REFERENCE_FILE, allow_pickle=False)
    common_x = fdm_reference["common_x"]
    common_y = fdm_reference["common_y"]
    ref_field = fdm_reference["final_field"]
    ref_times = fdm_reference["gauge_times"]
    ref_gauges = fdm_reference["gauge_values"]

    if RUN_FEM_REFERENCE:
        print(f"\nComputing FEM reference with N={REFERENCE_N} ...")
        fem_reference_result = solve_fem(REFERENCE_N, track_energy=True)
        save_fem_reference(fem_reference_result, fdm_reference)
        fem_reference_field = interpolate_field(
            fem_reference_result.x,
            fem_reference_result.y,
            fem_reference_result.final_field,
            common_x,
            common_y,
        )
        agreement_field = field_metrics(fem_reference_field, ref_field, common_x, common_y)
        agreement_gauges = gauge_metrics(
            fem_reference_result.times,
            fem_reference_result.gauge_values,
            ref_times,
            ref_gauges,
        )
        agreement_path = os.path.join(OUTPUT_DIR, "reference_agreement.csv")
        agreement_row = {
            "fdm_reference_N": int(fdm_reference["N"]),
            "fdm_reference_dt": float(fdm_reference["dt"]),
            "fem_reference_N": REFERENCE_N,
            "fem_reference_dt": fem_reference_result.dt,
            "relative_field_difference_T": agreement_field["relative_L2"],
            "Linf_field_difference_T": agreement_field["Linf"],
            "relative_H1_difference_T": agreement_field["relative_H1"],
            "maximum_gauge_waveform_difference": float(
                np.max(agreement_gauges["waveform_errors"])
            ),
            "mean_gauge_waveform_difference": agreement_gauges["mean_waveform_error"],
        }
        with open(agreement_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(agreement_row.keys()))
            writer.writeheader()
            writer.writerow(agreement_row)
        print(f"Saved reference agreement: {agreement_path}")
        print(
            f"Reference agreement: rel L2={agreement_row['relative_field_difference_T']:.6e}, "
            f"max gauge error={agreement_row['maximum_gauge_waveform_difference']:.6e}"
        )

    if not RUN_SCALABILITY:
        return

    warmup_N = max(6, min(LEVELS.values()) // 2)
    print(f"\nWarm-up run with N={warmup_N} ...")
    _ = solve_fem(warmup_N, track_energy=False, snapshot_times=[])

    rows = []
    representative_result: Optional[SolverResult] = None
    representative_metrics: Optional[Dict[str, float]] = None

    for level, N in LEVELS.items():
        print(f"\nFEM {level}: N={N}")
        measured_results = []
        for repeat in range(TIMING_REPEATS):
            result = solve_fem(
                N,
                track_energy=True,
                snapshot_times=SNAPSHOT_TIMES if repeat == 0 else [],
            )
            measured_results.append(result)
            print(
                f"  repeat {repeat + 1}/{TIMING_REPEATS}: "
                f"assembly={result.assembly_time_s:.3f} s, "
                f"solve={result.solve_time_s:.3f} s, "
                f"total={result.total_time_s:.3f} s, "
                f"RSS={result.peak_cpu_memory_mb:.2f} MB"
            )

        primary = measured_results[0]
        common_field = interpolate_field(
            primary.x, primary.y, primary.final_field, common_x, common_y
        )
        metrics = field_metrics(common_field, ref_field, common_x, common_y)
        gauge_info = gauge_metrics(primary.times, primary.gauge_values, ref_times, ref_gauges)
        energy_drift = float(
            np.max(np.abs(primary.energies - primary.energies[0]))
            / max(abs(primary.energies[0]), 1.0e-30)
        )

        assembly_times = np.array([item.assembly_time_s for item in measured_results])
        solve_times = np.array([item.solve_time_s for item in measured_results])
        total_times = np.array([item.total_time_s for item in measured_results])
        memories = np.array([item.peak_cpu_memory_mb for item in measured_results])

        row = {
            "method": "FEM",
            "level": level,
            "N": N,
            "unknowns": (N - 1) ** 2,
            "elements": 2 * N * N,
            "dt": primary.dt,
            "time_steps": primary.Nt,
            "relative_L2": metrics["relative_L2"],
            "Linf": metrics["Linf"],
            "relative_H1": metrics["relative_H1"],
            "mean_gauge_error": gauge_info["mean_waveform_error"],
            "maximum_arrival_time_error": gauge_info["maximum_arrival_error"],
            "energy_drift": energy_drift,
            "assembly_time_mean_s": float(np.mean(assembly_times)),
            "assembly_time_std_s": float(np.std(assembly_times, ddof=1)) if len(assembly_times) > 1 else 0.0,
            "solve_time_mean_s": float(np.mean(solve_times)),
            "solve_time_std_s": float(np.std(solve_times, ddof=1)) if len(solve_times) > 1 else 0.0,
            "total_time_mean_s": float(np.mean(total_times)),
            "total_time_std_s": float(np.std(total_times, ddof=1)) if len(total_times) > 1 else 0.0,
            "peak_cpu_memory_mean_MB": float(np.mean(memories)),
            "peak_cpu_memory_std_MB": float(np.std(memories, ddof=1)) if len(memories) > 1 else 0.0,
        }
        rows.append(row)

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

    csv_path = os.path.join(OUTPUT_DIR, "fem_scalability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved FEM scalability table: {csv_path}")

    if MAKE_PLOTS and representative_result is not None and representative_metrics is not None:
        make_fem_plots(representative_result, fdm_reference)

    print("\nFEM table values")
    for row in rows:
        print(
            f"{row['level']}: N={row['N']}, unknowns={row['unknowns']}, "
            f"Nt={row['time_steps']}, relL2={row['relative_L2']:.6e}, "
            f"time={row['total_time_mean_s']:.3f} +/- {row['total_time_std_s']:.3f} s, "
            f"memory={row['peak_cpu_memory_mean_MB']:.2f} +/- "
            f"{row['peak_cpu_memory_std_MB']:.2f} MB"
        )


if __name__ == "__main__":
    main()