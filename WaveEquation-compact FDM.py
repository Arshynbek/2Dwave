""" Fourth_Order_Wave_Section.
All experiments run to T=1 on the unit square, using float64 arithmetic.
The compact scheme uses h=1+x^2+y^2 
"""
import argparse
import csv
from pathlib import Path
from math import log2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.lapack import dpttrf, dpttrs

SPACE_GRIDS = (20, 40, 80, 160, 320)
TIME_STEPS = (1000, 2000, 4000, 8000)


def coefficient_second(values, spacing):
    """Fourth-order second derivative along the first axis on all interior nodes."""
    second = np.empty_like(values[1:-1])
    second[1:-1] = (-values[:-4] + 16 * values[1:-3] - 30 * values[2:-2]
                    + 16 * values[3:-1] - values[4:])
    edge = np.array([10., -15., -4., 14., -6., 1.])
    second[0] = edge @ values[:6]
    second[-1] = edge @ values[-6:][::-1]
    return second / (12 * spacing**2)


class NumerovLine:
    """Factor the fixed SPD line matrix, with diagonal 10/12 and off-diagonal 1/12."""

    def __init__(self, size, spacing):
        self.spacing = spacing
        self.diag, self.off, info = dpttrf(np.full(size, 10/12), np.full(size - 1, 1/12))
        if info != 0:
            raise ArithmeticError("Factorization of the Numerov matrix failed.")

    def solve(self, rhs):
        solution, info = dpttrs(self.diag, self.off, np.array(rhs, order="F", copy=True),
                                overwrite_b=True)
        if info != 0:
            raise ArithmeticError("Numerov line solve failed.")
        return solution

    def stiffness(self, values):
        rhs = 2 * values
        rhs[1:] -= values[:-1]
        rhs[:-1] -= values[1:]
        return self.solve(rhs) / self.spacing**2

    def boundary_load(self, lower, upper, interior_shape):
        rhs = np.zeros(interior_shape)
        rhs[0] += lower / 12
        rhs[-1] += upper / 12
        return self.solve(rhs)


class CompactWave:
    """Apply the symmetric spatial operator A and the boundary-corrected load."""

    def __init__(self, h, dx, dy, q=None):
        h = np.asarray(h, dtype=float)
        minimum_intervals = 5 if q is None else 3
        if h.ndim != 2 or min(h.shape) <= minimum_intervals:
            raise ValueError(f"Use at least {minimum_intervals} intervals per direction.")
        if not np.all(np.isfinite([dx, dy])) or dx <= 0 or dy <= 0:
            raise ValueError("Both spatial spacings must be finite and positive.")
        if not np.all(np.isfinite(h)) or np.min(h) <= 0:
            raise ValueError("The nodal coefficient must be finite and positive.")
        self.dx, self.dy = dx, dy
        self.s = np.sqrt(h)
        self.interior_s = self.s[1:-1, 1:-1]
        self.shape = self.interior_s.shape
        self.xline = NumerovLine(self.shape[0], dx)
        self.yline = NumerovLine(self.shape[1], dy)
        if q is None:
            lap_s = (coefficient_second(self.s, dx)[:, 1:-1]
                     + coefficient_second(self.s.T, dy).T[1:-1, :])
            self.q = lap_s / self.interior_s
        else:
            self.q = np.broadcast_to(np.asarray(q, dtype=float), self.shape).copy()
        if not np.all(np.isfinite(self.q)):
            raise ValueError("The transformed potential must be finite.")
        bound = np.max(h) * (6/dx**2 + 6/dy**2 + max(0., np.max(self.q)))
        self.cfl_limit = 2 / np.sqrt(bound)

    def apply(self, u):
        """A u, on the interior grid. Dirichlet values of u are zero."""
        w = self.interior_s * u
        stiffness = self.xline.stiffness(w) + self.yline.stiffness(w.T).T
        return self.interior_s * (stiffness + self.q * w)

    def load(self, f):
        """Map FULL nodal forcing, including its four side traces, to the interior."""
        if np.shape(f) != self.s.shape:
            raise ValueError("Provide forcing values at every node, including all sides.")
        g = np.asarray(f, dtype=float) / self.s
        bx = self.xline.boundary_load(g[0, 1:-1], g[-1, 1:-1], self.shape)
        by = self.yline.boundary_load(g[1:-1, 0], g[1:-1, -1], self.shape[::-1]).T
        return self.interior_s * (g[1:-1, 1:-1] + bx + by)


def leapfrog(operator, load_at, u0, v0, steps, record_energy=False):
    """Taylor start and centred time stepping to T=1; energy is evaluated at half steps."""
    dt = 1 / steps
    if dt >= operator.cfl_limit:
        raise ValueError("Increase the number of time steps to meet the sufficient CFL bound.")
    previous = u0.copy()
    a_previous = operator.apply(previous)
    current = previous + dt*v0 + 0.5*dt**2*(load_at(0.) - a_previous)
    energy = np.empty(steps) if record_energy else None
    for k in range(steps):
        if record_energy:
            velocity = (current - previous)/dt
            energy[k] = 0.5*operator.dx*operator.dy*(
                np.sum(velocity**2) + np.sum(a_previous*current))
        if k+1 < steps:
            a_current = operator.apply(current)
            following = 2*current - previous + dt**2*(load_at((k+1)*dt) - a_current)
            previous, current, a_previous = current, following, a_current
    return current, energy


def exponential_shape(z, exponent):
    """z(1-z)exp(exponent*z) and its first two derivatives."""
    base = z * (1-z)
    exp = np.exp(exponent*z)
    return (base*exp, (1-2*z + exponent*base)*exp,
            (-2 + 2*exponent*(1-2*z) + exponent**2*base)*exp)


def compact_run(n, case, steps):
    """Return final interior field, max error, discrete L2 error and max residual at t=0."""
    d = 1 / n
    nodes = np.linspace(0., 1., n+1)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    if case == "paper":
        h, hx, hy = 1+x*x+y*y, 2*x, 2*y
        px, py = np.sin(np.pi*x), np.sin(np.pi*y)
        px_x, py_y = np.pi*np.cos(np.pi*x), np.pi*np.cos(np.pi*y)
        lap = -2*np.pi**2*px*py
        time_factor, initial_speed, acceleration = np.cos, 0., -1.
    elif case == "asymmetric":
        h, hx, hy = 2+x+x*y+y*y, 1+y, x+2*y
        px, px_x, pxx = exponential_shape(x, 1.)
        py, py_y, pyy = exponential_shape(y, -0.5)
        lap = pxx*py + px*pyy
        time_factor, initial_speed, acceleration = lambda t: 1+t+t*t, 1., 2.
    else:
        raise ValueError("Unknown manufactured test.")
    phi = px*py
    div_flux = h*lap + hx*px_x*py + hy*px*py_y
    phi_i = phi[1:-1, 1:-1]
    op = CompactWave(h, d, d)
    if case == "paper":
        amplitude = op.load(-phi-div_flux)
        def load_at(t):
            return np.cos(t)*amplitude
    else:
        constant = op.load(2*phi)
        amplitude = op.load(-div_flux)
        def load_at(t):
            return constant + time_factor(t)*amplitude
    # Spatial residual includes the entire set of interior nodes at time zero.
    residual = acceleration*phi_i + op.apply(phi_i) - load_at(0.)
    computed, _ = leapfrog(op, load_at, phi_i, initial_speed*phi_i, steps)
    error = computed - time_factor(1.)*phi_i
    return computed, np.max(np.abs(error)), d*np.linalg.norm(error), np.max(np.abs(residual))


def figure_fields(n=40, steps=8000):
    """Benchmark plotted on the manuscript grid, including all half-step energies."""
    d, dt = 1 / n, 1 / steps
    nodes = np.linspace(0., 1., n+1)
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    h = 1 + x*x + y*y
    sx, sy = np.sin(np.pi*x), np.sin(np.pi*y)
    phi = sx * sy
    f0 = ((2*np.pi**2*h - 1)*phi
          - 2*np.pi*x*np.cos(np.pi*x)*sy - 2*np.pi*y*sx*np.cos(np.pi*y))
    operator = CompactWave(h, d, d)
    load = operator.load(f0)
    initial = phi[1:-1, 1:-1].copy()
    current, energy = leapfrog(operator, lambda t: np.cos(t)*load, initial,
                               np.zeros_like(initial), steps, record_energy=True)
    half_times = (np.arange(steps) + 0.5)*dt
    numerical = np.pad(current, 1)
    exact = np.cos(1.)*phi
    exact[[0, -1], :] = 0.
    exact[:, [0, -1]] = 0.
    return x, y, exact, numerical, half_times, energy


def exact_energy(t):
    return np.sin(t)**2 / 8 + 5*np.pi**2*np.cos(t)**2 / 12


def plot_fields(folder, x, y, exact, numerical):
    """Plot the numerical solution, exact solution and absolute error as 3D surfaces."""
    fig = plt.figure(figsize=(9.2, 3.05), facecolor="white")
    panels = [
        ("(a) Compact FDM", numerical, "viridis", 0.55, [0.25, 0.50], r"$u$"),
        ("(b) Exact solution", exact, "viridis", 0.55, [0.25, 0.50], r"$u$"),
        ("(c) Absolute error", 1e8*np.abs(numerical-exact), "inferno",
         9.0, [2, 4, 6, 8], r"$10^{-8}$"),
    ]
    for i, (title, values, cmap, upper, ticks, z_label) in enumerate(panels):
        left = 0.008 + i/3
        ax = fig.add_axes([left, 0.04, 0.295, 0.84], projection="3d")
        ax.view_init(elev=27, azim=-57)
        ax.set_box_aspect((1, 1, 0.78))
        ax.plot_surface(x, y, values, rstride=1, cstride=1, cmap=cmap, vmin=0,
                        vmax=upper, linewidth=0, antialiased=True, shade=False)
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, upper), xticks=[0, 0.5, 1],
               yticks=[0.5, 1], zticks=ticks)
        ax.set_xlabel(r"$x$", labelpad=-4)
        ax.set_ylabel(r"$y$", labelpad=-4)
        ax.text(1, 1, upper*1.07, z_label, ha="center", va="bottom", fontsize=10)
        ax.tick_params(labelsize=9, pad=-2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.965, 0.975, 0.98, 1))
        fig.text(left + 0.1475, 0.94, title, ha="center", fontsize=12)
    fig.savefig(folder / "FDM-t=1.0.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_energy(folder, times, energy):
    t = np.linspace(0, 1, 41)
    ix = np.linspace(0, len(times) - 1, min(17, len(times)), dtype=int)
    fig, ax = plt.subplots(figsize=(8,5), layout="constrained")
    ax.plot(times[ix], energy[ix], 'b-', fillstyle='none', markersize=4, label="Compact FDM")
    ax.plot(t, exact_energy(t),  'ro', fillstyle='none',  markersize=4, label="Exact")
    ax.set(xlabel=r"$t$", ylabel="Energy", xlim=(0, 1), ylim=(1.0, 4.35),  xticks=[0, 0.25, 0.50, 0.75, 1.0])
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(folder / "Energy_FDM.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return np.max(np.abs(energy - exact_energy(times)))

def face_derivative(values, d):
    """Nodal values to faces or face fluxes to interior nodes along axis 0."""
    result = np.empty((len(values)-1, values.shape[1]), dtype=values.dtype)
    result[0] = (-11/12*values[0] + 17/24*values[1] + 3/8*values[2]
                 - 5/24*values[3] + 1/24*values[4]) / d
    result[-1] = (11/12*values[-1] - 17/24*values[-2] - 3/8*values[-3]
                  + 5/24*values[-4] - 1/24*values[-5]) / d
    result[1:-1] = (values[:-3] - 27*values[1:-2] + 27*values[2:-1] - values[3:]) / (24*d)
    return result


def staggered_run(n, steps):
    """Staggered reference calculation with h=x^2+y^2 and its matching forcing."""
    d, dt = 1/n, 1/steps
    nodes = np.linspace(0., 1., n+1)
    faces = (np.arange(n)+0.5)*d
    x, y = np.meshgrid(nodes, nodes, indexing="ij")
    sx, sy = np.sin(np.pi*x), np.sin(np.pi*y)
    phi = sx*sy
    grad_term = 2*np.pi*(x*sy*np.cos(np.pi*x) + y*sx*np.cos(np.pi*y))
    lap_term = -2*np.pi**2*(x**2+y**2)*phi
    f0 = -(phi + grad_term + lap_term)
    h_xface = faces[:, None]**2 + nodes[None, :]**2
    h_yface = nodes[:, None]**2 + faces[None, :]**2

    def divergence(u):
        fx = h_xface*face_derivative(u, d)
        fy = h_yface*face_derivative(u.T, d).T
        return face_derivative(fx, d)[:, 1:-1] + face_derivative(fy.T, d).T[1:-1, :]

    def apply_boundary(u, t):
        # Evaluate the same analytical Dirichlet trace as the reference implementation.
        amplitude = np.cos(t)
        u[[0, -1], :] = amplitude*sx[[0, -1], :]*sy[[0, -1], :]
        u[:, [0, -1]] = amplitude*sx[:, [0, -1]]*sy[:, [0, -1]]

    interior = np.s_[1:-1, 1:-1]
    previous = phi.copy()
    current = previous.copy()
    current[interior] += 0.5*dt*dt*(divergence(previous) + f0[interior])
    t = dt
    apply_boundary(current, t)
    for _ in range(1, steps):
        previous[interior] = (2*current[interior] - previous[interior]
                              + dt*dt*(divergence(current) + np.cos(t)*f0[interior]))
        t += dt
        apply_boundary(previous, t)
        previous, current = current, previous
    error = current - np.cos(1.)*sx*sy
    return np.max(np.abs(error)), np.sqrt(np.sum(error**2)*d*d)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def latex_number(value, digits):
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"$ {mantissa}\times 10^{{{int(exponent)}}} $"


def write_table(folder, name, caption, label, axis, panels):
    """Each panel consists of its title, ordered (N,Nt,Linf,L2) rows and precision."""
    step_heading = r"$\Delta x=\Delta y$" if axis == "space" else r"$\Delta t$"
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{" + caption + "}",
             r"\label{" + label + "}", r"\small", r"\begin{tabular}{crcrc}", r"\hline",
             step_heading + r" & $\|e\|_\infty$ & $\mathrm{CO}_{L^\infty}$"
             + r" & $\|e\|_c$ & $\mathrm{CO}_{L^2}$\\", r"\hline"]
    csv_rows = []
    for title, data, digits in panels:
        if title:
            lines.append(r"\multicolumn{5}{l}{\emph{" + title + r"}}\\")
        previous = None
        for n, steps, linf, l2 in data:
            divisions = n if axis == "space" else steps
            if previous is None:
                pmax = p2 = None
            else:
                old_div, old_max, old_l2 = previous
                pmax = log2(old_max/linf) / log2(divisions/old_div)
                p2 = log2(old_l2/l2) / log2(divisions/old_div)
            csv_rows.append((n, steps, 1/n, 1/steps, linf, pmax, l2, p2))
            orders = ["--" if p is None else f"{p:.3f}" for p in (pmax, p2)]
            lines.append(" & ".join([f"$1/{divisions}$", latex_number(linf, digits),
                                     orders[0], latex_number(l2, digits), orders[1]]) + r"\\")
            previous = (divisions, linf, l2)
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    (folder / (name+".tex")).write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_csv(folder / (name+".csv"), ("N", "Nt", "dx=dy", "dt", "Linf", "CO_Linf", "L2", "CO_L2"), csv_rows)


def compact_tables(folder):
    panels = []
    residuals = []
    for steps in (8000, 16000):
        rows = []
        for n in SPACE_GRIDS:
            _, linf, l2, residual = compact_run(n, "paper", steps)
            rows.append((n, steps, linf, l2))
            print(f"Compact spatial  N={n:3d}  Nt={steps}  L2={l2:.6e}", flush=True)
            if steps == 8000:
                order = log2(residuals[-1][1]/residual) if residuals else None
                residuals.append((n, residual, order))
        panels.append((rf"Fixed $\Delta t=1/{steps}$", rows, 6 if steps == 8000 else 5))
    write_table(folder, "table1_compact_space", "Compact spatial refinement at $T=1$ for $h=1+x^2+y^2$.",
                "tab:cw-paper", "space", panels)
    write_csv(folder / "spatial_residual.csv", ("N", "max_residual_t0", "CO"), residuals)

    rows, differences = [], []
    previous_field = None
    for steps in TIME_STEPS:
        field, linf, l2, _ = compact_run(300, "paper", steps)
        if previous_field is not None:
            difference = np.linalg.norm(previous_field-field)/300
            order = log2(differences[-1][2]/difference) if differences else None
            differences.append((steps//2, steps, difference, order))
        rows.append((300, steps, linf, l2))
        previous_field = field
        print(f"Compact temporal N=300  Nt={steps}  L2={l2:.6e}", flush=True)
    write_table(folder, "table2_compact_time", "Compact temporal refinement at $T=1$ for $N=300$.",
                "tab:cw-time", "time", [("", rows, 6)])
    write_csv(folder / "temporal_self_convergence.csv", ("Nt_coarse", "Nt_fine", "L2_difference", "CO"), differences)

    boundary = []
    for n in SPACE_GRIDS[:-1]:
        _, linf, l2, _ = compact_run(n, "asymmetric", 8000)
        order = log2(boundary[-1][3]/l2) if boundary else None
        boundary.append((n, 8000, linf, l2, order))
    write_csv(folder / "test2_boundary_load.csv", ("N", "Nt", "Linf", "L2", "CO_L2"), boundary)


def staggered_tables(folder):
    """Compute final-time errors for the spatial and temporal refinement studies."""
    spatial_rows = []
    for n in SPACE_GRIDS:
        linf, l2 = staggered_run(n, 8000)
        spatial_rows.append((n, 8000, linf, l2))
        print(f"Staggered spatial  N={n}  Nt=8000  L2={l2:.6e}", flush=True)
    write_table(folder, "table3_staggered_space",
                r"Staggered-grid spatial refinement at $T=1$ for $h_0=x^2+y^2$ and $\Delta t=1/8000$.",
                "tab1", "space", [("", spatial_rows, 6)])

    temporal_rows = []
    for steps in TIME_STEPS:
        linf, l2 = staggered_run(300, steps)
        temporal_rows.append((300, steps, linf, l2))
        print(f"Staggered temporal N=300  Nt={steps}  L2={l2:.6e}", flush=True)
    write_table(folder, "table4_staggered_time",
                r"Staggered-grid temporal refinement at $T=1$ for $h_0=x^2+y^2$ and $\Delta x=\Delta y=1/300$.",
                "tab2", "time", [("", temporal_rows, 6)])


def make_figures(folder):
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "axes.labelsize": 11, "axes.linewidth": 0.75, "savefig.facecolor": "white"})
    x, y, exact, numerical, times, energy = figure_fields()
    plot_fields(folder, x, y, exact, numerical)
    energy_error = plot_energy(folder, times, energy)
    write_csv(folder / "energy_half_steps.csv",
              ("t_half", "E_numerical", "E_exact", "absolute_error"),
              zip(times, energy, exact_energy(times), np.abs(energy-exact_energy(times))))
    print(f"Figures: max displacement error {np.max(np.abs(numerical-exact)):.6e}; "
          f"max energy difference {energy_error:.6e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["all", "figures", "compact", "staggered"], default="all")
    parser.add_argument("--out", type=Path, default=Path("wave_results"), help="output directory")
    args = parser.parse_args()
    if args.only in ("all", "figures"):
        figures = args.out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        make_figures(figures)
    if args.only in ("all", "compact", "staggered"):
        tables = args.out / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        if args.only in ("all", "compact"):
            compact_tables(tables)
        if args.only in ("all", "staggered"):
            staggered_tables(tables)
    print(f"Output directory: {args.out.resolve()}")


if __name__ == "__main__":
    main()
