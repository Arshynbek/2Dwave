# ============================================================
# Model problem:
# u_tt - div(h ∇u) = f on (t,x,y) in [0,T] × [0,1]^2
# IC and BC:
#   u(0,x,y) = u0(x,y),
#   u_t(0,x,y) = u1(x,y) (here u1 ≡ 0),
#   u|_{∂Ω} = 0.
# Exact solution: u(t,x,y) = cos(t) sin(πx) sin(πy),
# h(x,y) = x^2 + y^2,   f chosen accordingly.
# ============================================================



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import timeit
import tracemalloc

tracemalloc.start()  # start tracing
start = timeit.default_timer()
# ---------- Exact data (given) ----------
def exact_solution(t, X, Y):
    return np.cos(t) * np.sin(np.pi * X) * np.sin(np.pi * Y)

def f_func(t, x, y):
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    term_g = 2 * np.pi * (x * np.sin(np.pi * y) * np.cos(np.pi * x)
                          + y * np.sin(np.pi * x) * np.cos(np.pi * y))
    term_l = -2 * (np.pi ** 2) * (x ** 2 + y ** 2) * S
    return -np.cos(t) * (S + term_g + term_l)

# ---- NEW: exact energy for comparison ----
def exact_energy(t):
    return 0.125 * np.sin(t)**2 + (np.pi**2 / 6.0) * np.cos(t)**2

# ---------- Grids ----------
def build_grids(Nx, Ny, Lx=1.0, Ly=1.0):
    if Nx < 6 or Ny < 6:
        raise ValueError("Use Nx, Ny >= 6 for 4th-order boundary stencils.")
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    Xc, Yc = np.meshgrid(x, y, indexing='ij')         # centers: (Nx+1, Ny+1)

    xf = (np.arange(Nx) + 0.5) * dx                   # vertical faces x
    Xfx, Yfx = np.meshgrid(xf, y, indexing='ij')      # (Nx, Ny+1)

    yf = (np.arange(Ny) + 0.5) * dy                   # horizontal faces y
    Xfy, Yfy = np.meshgrid(x, yf, indexing='ij')      # (Nx+1, Ny)
    return dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy

# One-sided 4th-order coeffs (forward) for derivative near boundary
c_os = np.array([-11/12, 17/24, 3/8, -5/24, 1/24])

# ---------- 4th-order MAC building blocks ----------
def grad_x_faces(u, dx):
    """(du/dx) at vertical faces (i+1/2, j). u at centers (Nx+1, Ny+1) -> (Nx, Ny+1)."""
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    gx = np.zeros((Nx, Ny+1), dtype=u.dtype)
    # left face (one-sided)
    gx[0,:] = (c_os[0]*u[0,:] + c_os[1]*u[1,:] + c_os[2]*u[2,:]
               + c_os[3]*u[3,:] + c_os[4]*u[4,:]) / dx
    # interior faces (central 4th-order at face)
    for i in range(1, Nx-1):
        gx[i,:] = (u[i-1,:] - 27*u[i,:] + 27*u[i+1,:] - u[i+2,:])/(24*dx)
    # right face (one-sided, mirrored)
    gx[Nx-1,:] = (+11/12*u[Nx,:] - 17/24*u[Nx-1,:] - 3/8*u[Nx-2,:]
                  + 5/24*u[Nx-3,:] - 1/24*u[Nx-4,:]) / dx
    return gx

def grad_y_faces(u, dy):
    """(du/dy) at horizontal faces (i, j+1/2). u at centers -> (Nx+1, Ny)."""
    Nx, Ny = u.shape[0]-1, u.shape[1]-1
    gy = np.zeros((Nx+1, Ny), dtype=u.dtype)
    # bottom face
    gy[:,0] = (c_os[0]*u[:,0] + c_os[1]*u[:,1] + c_os[2]*u[:,2]
               + c_os[3]*u[:,3] + c_os[4]*u[:,4]) / dy
    # interior faces
    for j in range(1, Ny-1):
        gy[:,j] = (u[:,j-1] - 27*u[:,j] + 27*u[:,j+1] - u[:,j+2])/(24*dy)
    # top face
    gy[:,Ny-1] = (+11/12*u[:,Ny] - 17/24*u[:,Ny-1] - 3/8*u[:,Ny-2]
                  + 5/24*u[:,Ny-3] - 1/24*u[:,Ny-4]) / dy
    return gy

def div_x_from_faces(Fx, dx):
    """∂_x Fx at centers from vertical-face Fx. Fx: (Nx, Ny+1) -> (Nx+1, Ny+1) (interior filled)."""
    Nx, Ny = Fx.shape[0], Fx.shape[1]-1
    d = np.zeros((Nx+1, Ny+1), dtype=Fx.dtype)
    # first interior center (i=1) one-sided
    d[1,:] = (-11/12*Fx[0,:] + 17/24*Fx[1,:] + 3/8*Fx[2,:] - 5/24*Fx[3,:] + 1/24*Fx[4,:]) / dx
    # interior centers (i=2..Nx-2) central
    for i in range(2, Nx-1):
        d[i,:] = (Fx[i-2,:] - 27*Fx[i-1,:] + 27*Fx[i,:] - Fx[i+1,:])/(24*dx)
    # last interior center (i=Nx-1) one-sided
    d[Nx-1,:] = (+11/12*Fx[Nx-1,:] - 17/24*Fx[Nx-2,:] - 3/8*Fx[Nx-3,:]
                 + 5/24*Fx[Nx-4,:] - 1/24*Fx[Nx-5,:]) / dx
    return d

def div_y_from_faces(Fy, dy):
    """∂_y Fy at centers from horizontal-face Fy. Fy: (Nx+1, Ny) -> (Nx+1, Ny+1) (interior filled)."""
    Nx, Ny = Fy.shape[0]-1, Fy.shape[1]
    d = np.zeros((Nx+1, Ny+1), dtype=Fy.dtype)
    # first interior center (j=1)
    d[:,1] = (-11/12*Fy[:,0] + 17/24*Fy[:,1] + 3/8*Fy[:,2] - 5/24*Fy[:,3] + 1/24*Fy[:,4]) / dy
    # interior centers (j=2..Ny-2)
    for j in range(2, Ny-1):
        d[:,j] = (Fy[:,j-2] - 27*Fy[:,j-1] + 27*Fy[:,j] - Fy[:,j+1])/(24*dy)
    # last interior center (j=Ny-1)
    d[:,Ny-1] = (+11/12*Fy[:,Ny-1] - 17/24*Fy[:,Ny-2] - 3/8*Fy[:,Ny-3]
                 + 5/24*Fy[:,Ny-4] - 1/24*Fy[:,Ny-5]) / dy
    return d

def L4_MAC(u, dx, dy, Xfx, Yfx, Xfy, Yfy):
    """
    4th-order staggered operator L u = div(h grad u).
    h(x,y)=x^2+y^2 evaluated *exactly at faces*.
    u: (Nx+1, Ny+1) at centers.
    """
    hfx = Xfx**2 + Yfx**2              # (Nx, Ny+1)
    hfy = Xfy**2 + Yfy**2              # (Nx+1, Ny)
    gx = grad_x_faces(u, dx)           # (Nx, Ny+1)
    gy = grad_y_faces(u, dy)           # (Nx+1, Ny)
    Fx = hfx * gx
    Fy = hfy * gy
    return div_x_from_faces(Fx, dx) + div_y_from_faces(Fy, dy)

def enforce_dirichlet(u, t, Xc, Yc):
    u[0,:]  = exact_solution(t, Xc[0,:],  Yc[0,:])
    u[-1,:] = exact_solution(t, Xc[-1,:], Yc[-1,:])
    u[:,0]  = exact_solution(t, Xc[:,0],  Yc[:,0])
    u[:,-1] = exact_solution(t, Xc[:,-1], Yc[:,-1])

# ---- NEW: numerical energy using MAC face-gradients + leapfrog velocity ----
def numerical_energy(u_prev, u_curr, u_next, t, dt, dx, dy, Xc, Yc, Xfx, Yfx, Xfy, Yfy):
    """
    Discrete energy:
      E_h = 1/2 [ sum (u_t)^2  + sum H (|u_x|_faces^2 + |u_y|_faces^2) ] * dx*dy
    u_t at time t is approximated by central difference (u_next - u_prev)/(2 dt).
    """
    # kinetic term at centers
    if u_next is not None and u_prev is not None:
        u_t = (u_next - u_prev) / (2.0 * dt)
    elif u_next is None and u_prev is not None:
        # last time level: one-sided (first order)
        u_t = (u_curr - u_prev) / dt
    elif u_prev is None and u_next is not None:
        # first time level: one-sided (first order)
        u_t = (u_next - u_curr) / dt
    else:
        # t=0 and known u_t(0)=0 for this benchmark
        u_t = np.zeros_like(u_curr)

    # gradient terms via MAC face derivatives
    gx = grad_x_faces(u_curr, dx)   # (Nx, Ny+1)
    gy = grad_y_faces(u_curr, dy)   # (Nx+1, Ny)
    hfx = Xfx**2 + Yfx**2
    hfy = Xfy**2 + Yfy**2

    term_kin = 0.5 * np.sum(u_t**2) * dx * dy
    term_gx  = 0.5 * np.sum(hfx * gx**2) * dx * dy
    term_gy  = 0.5 * np.sum(hfy * gy**2) * dx * dy
    return term_kin + term_gx + term_gy

# ---------- Time stepping: explicit leapfrog (2nd order) ----------
def wave_solver_mac4(Nx=40, Ny=40, T=0.1, dt=1/4000, Lx=1.0, Ly=1.0, track_energy=True):
    dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy = build_grids(Nx, Ny, Lx, Ly)
    Nt = int(np.round(T/dt))
    dt = T / Nt  # snap to end time

    # Initial conditions
    u0 = exact_solution(0.0, Xc, Yc)
    u_t0 = np.zeros_like(u0)  # u_t(0)=0 for cos(t)*...
    enforce_dirichlet(u0, 0.0, Xc, Yc)

    # Taylor start (uses PDE at t=0)
    L_u0 = L4_MAC(u0, dx, dy, Xfx, Yfx, Xfy, Yfy)
    f0 = f_func(0.0, Xc, Yc)
    u_prev = u0
    u_curr = u0 + dt*u_t0 + 0.5*dt*dt*(L_u0 + f0)
    enforce_dirichlet(u_curr, dt, Xc, Yc)

    # ---- NEW: energy tracking arrays ----
    times, E_num, E_ex = [], [], []
    if track_energy:
        # E at t=0 (use exact u_t(0)=0)
        E0 = numerical_energy(u_prev=None, u_curr=u_prev, u_next=u_curr, t=0.0,
                              dt=dt, dx=dx, dy=dy, Xc=Xc, Yc=Yc, Xfx=Xfx, Yfx=Yfx, Xfy=Xfy, Yfy=Yfy)
        times.append(0.0); E_num.append(E0); E_ex.append(exact_energy(0.0))

    # Leapfrog
    t = dt
    for n in range(1, Nt):
        L_uc = L4_MAC(u_curr, dx, dy, Xfx, Yfx, Xfy, Yfy)
        f_n  = f_func(t, Xc, Yc)
        u_next = 2*u_curr - u_prev + dt*dt*(L_uc + f_n)
        t_next = t + dt
        enforce_dirichlet(u_next, t_next, Xc, Yc)

        # ---- NEW: record energy at the *current* time t using central u_t ----
        if track_energy:
            En = numerical_energy(u_prev, u_curr, u_next, t, dt, dx, dy, Xc, Yc, Xfx, Yfx, Xfy, Yfy)
            times.append(t); E_num.append(En); E_ex.append(exact_energy(t))

        u_prev, u_curr = u_curr, u_next
        t = t_next

    # Record final energy at T (use one-sided velocity)
    if track_energy and abs(t - T) < 1e-14:
        EnT = numerical_energy(u_prev, u_curr, None, T, dt, dx, dy, Xc, Yc, Xfx, Yfx, Xfy, Yfy)
        times.append(T); E_num.append(EnT); E_ex.append(exact_energy(T))

    # Errors at T
    u_exact_T = exact_solution(T, Xc, Yc)
    err = u_curr - u_exact_T
    max_err = np.max(np.abs(err))
    L2_err = np.sqrt(np.sum(err**2) * dx * dy)

    out = (Xc, Yc, u_curr, u_exact_T, err, max_err, L2_err)
    if track_energy:
        out = out + (np.array(times), np.array(E_num), np.array(E_ex))
    return out

# ------------- Example run + visualization -------------
if __name__ == "__main__":
    Nx = Ny = 40
    T  = 1.0
    dt = 1/80

    results = wave_solver_mac4(Nx=Nx, Ny=Ny, T=T, dt=dt, track_energy=True)
    (X, Y, u_FDM, u_exact_T, err, max_err, L2_err, times, E_num, E_ex) = results

    print(f"N={Nx}, T={T}, dt={dt}")
    print(f"Max error = {max_err:.6e}")
    print(f"L2  error = {L2_err:.6e}")

    # ----- Visualization: solutions and energy -----
    fig = plt.figure(figsize=(18, 5))

    # Numerical solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, u_FDM, cmap='viridis', rstride=1, cstride=1)
    ax1.set_title('Numerical Solution at T')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

    # Exact solution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_exact_T, cmap='viridis', rstride=1, cstride=1)
    ax2.set_title('Exact Solution at T')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

    # Error distribution (absolute)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, np.abs(err), cmap='inferno', rstride=1, cstride=1)
    ax3.set_title('Error |u_num - u_exact|')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('abs error')

    plt.tight_layout()
    plt.show()

    # ---- NEW: Energy comparison plot ----
    plt.figure(figsize=(8,5))
    plt.plot(times, E_num,  'b-', fillstyle='none',  markersize=4, label='Numerical energy $E_h(t)$')
    plt.plot(times, E_ex,  'ro', fillstyle='none',  markersize=4, label='Exact energy $E_{\\mathrm{exact}}(t)$')
    plt.xlabel('t'); plt.ylabel('Energy')
    plt.title('Energy (numerical vs exact)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (Optional) relative energy error
    rel_err = np.abs(np.array(E_num) - np.array(E_ex)) / np.maximum(1e-14, np.array(E_ex))
    print(f"Energy rel. error at T: {rel_err[-1]:.3e}")
    stop = timeit.default_timer()
    print('Time: ', stop - start)


    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024**2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
    tracemalloc.stop()