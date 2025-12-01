# Single-case runner with choice + animation (GIF) and final surface plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from matplotlib import cm
import timeit
import tracemalloc

tracemalloc.start()
start = timeit.default_timer()


# -------------------------
# Piecewise h(x) (Cases 1 & 2)
# -------------------------
def h_case_vals(x, case_id):
    """
    x: array of x-locations (faces)
    case_id: 1 or 2
    returns h(x) with broadcasting over y.
    """
    if case_id == 1:
        # Case 1
        h = np.where(x < 0.75, 1.0, 0.1)
    elif case_id == 2:
        # Case 2
        h = np.where(x < 0.3, 1.0,
                     np.where(x < 0.8, 1.0 - 1.8*(x - 0.3), 0.1))
    else:
        raise ValueError("Unknown case_id (use 1 or 2).")
    return h

# -------------------------
# Grids
# -------------------------
def build_grids(Nx, Ny, Lx=1.0, Ly=1.0):
    assert Nx >= 6 and Ny >= 6, "Need at least 6 cells per direction for 4th-order one-sided stencils."
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(0, Lx, Nx+1)            # cell centers in x
    y = np.linspace(0, Ly, Ny+1)            # cell centers in y
    Xc, Yc = np.meshgrid(x, y, indexing='ij')         # centers: (Nx+1, Ny+1)

    xf = (np.arange(Nx) + 0.5) * dx         # vertical faces (x at i+1/2)
    Xfx, Yfx = np.meshgrid(xf, y, indexing='ij')      # (Nx, Ny+1)

    yf = (np.arange(Ny) + 0.5) * dy         # horizontal faces (y at j+1/2)
    Xfy, Yfy = np.meshgrid(x, yf, indexing='ij')      # (Nx+1, Ny)

    return dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy

# One-sided 4th-order coeffs (forward) for derivative near boundary
c_os = np.array([-11/12, 17/24, 3/8, -5/24, 1/24])

# -------------------------
# 4th-order MAC building blocks
# -------------------------
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
    # first interior center (one-sided)
    d[1,:] = (-11/12*Fx[0,:] + 17/24*Fx[1,:] + 3/8*Fx[2,:] - 5/24*Fx[3,:] + 1/24*Fx[4,:]) / dx
    # interior centers (central)
    for i in range(2, Nx-1):
        d[i,:] = (Fx[i-2,:] - 27*Fx[i-1,:] + 27*Fx[i,:] - Fx[i+1,:])/(24*dx)
    # last interior center (one-sided)
    d[Nx-1,:] = (+11/12*Fx[Nx-1,:] - 17/24*Fx[Nx-2,:] - 3/8*Fx[Nx-3,:]
                 + 5/24*Fx[Nx-4,:] - 1/24*Fx[Nx-5,:]) / dx
    return d

def div_y_from_faces(Fy, dy):
    """∂_y Fy at centers from horizontal-face Fy. Fy: (Nx+1, Ny) -> (Nx+1, Ny+1) (interior filled)."""
    Nx, Ny = Fy.shape[0]-1, Fy.shape[1]
    d = np.zeros((Nx+1, Ny+1), dtype=Fy.dtype)
    # first interior center
    d[:,1] = (-11/12*Fy[:,0] + 17/24*Fy[:,1] + 3/8*Fy[:,2] - 5/24*Fy[:,3] + 1/24*Fy[:,4]) / dy
    # interior centers
    for j in range(2, Ny-1):
        d[:,j] = (Fy[:,j-2] - 27*Fy[:,j-1] + 27*Fy[:,j] - Fy[:,j+1])/(24*dy)
    # last interior center
    d[:,Ny-1] = (+11/12*Fy[:,Ny-1] - 17/24*Fy[:,Ny-2] - 3/8*Fy[:,Ny-3]
                 + 5/24*Fy[:,Ny-4] - 1/24*Fy[:,Ny-5]) / dy
    return d


def L4_MAC(u, dx, dy, Xfx, Yfx, Xfy, Yfy, case_id):
    """
    4th-order staggered operator L u = div(h grad u).
    h = h(x) according to given case; evaluated at faces.
    u: (Nx+1, Ny+1) at centers.
    """
    # h at faces (depends only on x)
    hfx = h_case_vals(Xfx, case_id)         # (Nx, Ny+1)
    hfy = h_case_vals(Xfy, case_id)         # (Nx+1, Ny)
    # face gradients
    gx = grad_x_faces(u, dx)                # (Nx, Ny+1)
    gy = grad_y_faces(u, dy)                # (Nx+1, Ny)
    # fluxes
    Fx = hfx * gx
    Fy = hfy * gy
    # divergence back to centers
    return div_x_from_faces(Fx, dx) + div_y_from_faces(Fy, dy)

# -------------------------
# Boundary + Initial data
# -------------------------
def enforce_dirichlet_zero(u):
    """Homogeneous Dirichlet boundary conditions u=0 on ∂Ω."""
    u[0,:]  = 0.0
    u[-1,:] = 0.0
    u[:,0]  = 0.0
    u[:,-1] = 0.0

def gaussian_initial(Xc, Yc):
    # u(0,x,y) = exp( -(x-0.5)^2/0.005 - (y-0.5)^2/0.005 )
    return 2*np.exp(-((Xc-0.5)**2)/0.002 - ((Yc-0.5)**2)/0.002)

# -------------------------
# Time stepping (explicit leapfrog, f=0, u_t(0)=0)
# -------------------------
def wave_solver_mac4_piecewise(Nx=320, Ny=320, T=0.5, CFL=0.3, case_id=1, Lx=1.0, Ly=1.0, store_every=2):
    dx, dy, x, y, Xc, Yc, Xfx, Yfx, Xfy, Yfy = build_grids(Nx, Ny, Lx, Ly)

    # Max h for a conservative (very simple) CFL
    h_max = 1.0  # both cases have max(h)=1.0
    dt = CFL * min(dx, dy) / np.sqrt(h_max)
    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt  # snap to exact end time

    # initial data
    u0   = gaussian_initial(Xc, Yc)
    enforce_dirichlet_zero(u0)

    # Taylor start using PDE: u_tt = div(h grad u) (since f=0)
    L_u0 = L4_MAC(u0, dx, dy, Xfx, Yfx, Xfy, Yfy, case_id)
    u_prev = u0
    u_curr = u0 + 0.5*(dt**2)*L_u0
    enforce_dirichlet_zero(u_curr)

    # storage for animation
    frames = [u_prev.copy()]
    t = dt
    for n in range(1, Nt):
        L_uc = L4_MAC(u_curr, dx, dy, Xfx, Yfx, Xfy, Yfy, case_id)
        u_next = 2*u_curr - u_prev + (dt**2)*L_uc
        enforce_dirichlet_zero(u_next)

        u_prev, u_curr = u_curr, u_next
        if n % store_every == 0 or n == Nt-1:
            frames.append(u_curr.copy())
        t += dt

    return (x, y, Xc, Yc, u0, u_curr, frames, dt, Nt, T, case_id)

# -------------------------
# Plot helpers
# -------------------------

def plot_u0_and_H(Xc, Yc, u0, case_id):
    # H at centers for visualization
    Hc = h_case_vals(Xc, case_id)
    fig0 = plt.figure(figsize=(10, 8))
    ax1 = fig0.subplots(subplot_kw={"projection": "3d"})
    ax1.plot_surface(Xc, Yc, -Hc, cmap='inferno', rstride=1, cstride=1)
    ax1.plot_surface(Xc, Yc, u0,  cmap='viridis', rstride=1, cstride=1)
    ax1.set_title('U0 and H')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u0 / -H')
    ax1.view_init(elev=30, azim=135)
    plt.tight_layout()

    plt.show()

def plot_surface_final(x, y, U, case_id, title):

    X, Y = np.meshgrid(x, y, indexing='ij')
    Hc = h_case_vals(X, case_id)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(X, Y, U, cmap=cm.jet, rstride=1, cstride=1) #linewidth=0, antialiased=True , cmap=cm.rainbow
    surf = ax.plot_surface(X, Y, U, cmap='viridis', edgecolor = 'black', linewidth = 0.2,   rstride=1, cstride=1)  # Slight transparency) # or 'jet'

    #ax.plot_wireframe(X, Y, U, color='blue', alpha=0.6)
    fig.colorbar(surf,ax=ax, shrink=0.5, aspect=5)
    #ax.plot_surface(X, Y, -Hc, cmap='inferno', rstride=1, cstride=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(t,x,y)')
    ax.set_title(title)
    ax.view_init(elev=30, azim=110)
    plt.show()

def animate_surface_fast(x, y, frames, case_id, stride_frames=3, rstride=1, cstride=1, interval_ms=60):
    X, Y = np.meshgrid(x, y, indexing='ij')
    frames_ds = frames[::max(1, stride_frames)]
    vmax = max(np.max(np.abs(f)) for f in frames_ds) + 1e-12
    zmin, zmax = -vmax, vmax

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=30, azim=135)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u(x,y,t)')
    ax.set_title(f"Wave propagation (surface) — Case {case_id}")

    surf = ax.plot_surface(X, Y, frames_ds[0], cmap='viridis',  edgecolor = 'black', linewidth = 0.2, rstride=rstride, cstride=cstride)

    def update(idx):
        nonlocal surf
        surf.remove()
        surf = ax.plot_surface(X, Y, frames_ds[idx], cmap='viridis', edgecolor = 'black', linewidth = 0.2, rstride=rstride, cstride=cstride) #cmap='viridis'
        ax.set_title(f"Wave propagation — Case {case_id} (frame {idx+1}/{len(frames_ds)})")
        return (surf,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames_ds), interval=interval_ms, blit=False)
    out_path = Path("wave_case1_surface.gif")
    try:
        anim.save(out_path.as_posix(), writer=animation.PillowWriter(fps=int(1000/max(interval_ms,1))))
        saved = True
    except Exception as e:
        print("Animation save failed:", e)
        saved = False
    plt.close(fig)
    return out_path, saved, len(frames_ds)


# >>> Choose ONE case to run <<<
# Set SELECT_CASE = 1 or 2
# -------------------------
SELECT_CASE = 1  # change to 2 to run the second case only
# Solver parameters
T  = 0.0
Nx = Ny = 160
CFL = 0.25
STORE_EVERY = 1  # keep every 2nd step for animation

# Run the chosen case
x, y, Xc, Yc, u0, U_FDM, frames, dt, Nt, T_end, case_id = wave_solver_mac4_piecewise(
    Nx=Nx, Ny=Ny, T=T, CFL=CFL, case_id=SELECT_CASE, store_every=STORE_EVERY
)

print(f"Case {SELECT_CASE}: Nx={Nx}, Ny={Ny}, T={T_end:.4f}, dt≈{dt:.4e}, Nt={Nt}, stored frames={len(frames)}")


# ---- Pre-animation plot: U0 and H ----
#plot_u0_and_H(Xc, Yc, u0, case_id)

# Final surface
plot_surface_final(x, y, U_FDM, case_id, title=f"FDM at T={T} (case={SELECT_CASE})")
# gif_path, ok, kept = animate_surface_fast(x, y, frames, case_id=1, stride_frames=3, rstride=1, cstride=1, interval_ms=80)
# print("Saved:", ok, "frames kept:", kept, "->", gif_path)# -------------------------
#
# #
# report timing + memory
print("dx=", dt, "maximum U :", np.max(U_FDM))
stop = timeit.default_timer()
print('Time: ', stop - start)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")
tracemalloc.stop()
