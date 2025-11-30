import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import timeit
import tracemalloc
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle

tracemalloc.start()  # start tracing

# -------------------------------------------------
# Model problem:
#   u_tt - div( h(x,y) ∇u ) = f  in (0,T]×(0,1)^2
#   u(0,x,y) = u0(x,y),   u_t(0,x,y) = 0,
#   u = 0 on ∂Ω.
# Exact solution: u(t,x,y) = cos(t) sin(πx) sin(πy),
# h(x,y) = x^2 + y^2,   f chosen accordingly.
# -------------------------------------------------
start = timeit.default_timer()

# -------------------------------
# Exact solution, forcing, coeff
# -------------------------------
def exact_u(t, x, y):
    return np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_grad_u(t, x, y):
    ux = np.cos(t) * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uy = np.cos(t) * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return ux, uy

def f(t, x, y):
    """
    Forcing chosen so that exact_u solves
      u_tt - div(h ∇u) = f
    with h(x,y) = x^2 + y^2.
    """
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    term_g = 2 * np.pi * (x * np.sin(np.pi * y) * np.cos(np.pi * x)
                          + y * np.sin(np.pi * x) * np.cos(np.pi * y))
    term_l = -2 * (np.pi ** 2) * (x ** 2 + y ** 2) * S
    return -np.cos(t) * (S + term_g + term_l)

def h_coef(x, y):
    """Variable coefficient h(x,y) = x^2 + y^2."""
    return x**2 + y**2

# Exact energy for comparison:
#   E_exact(t) = 1/2 ∫ (u_t^2 + h|∇u|^2) dx dy
# for the chosen exact solution.
def exact_energy(t):
    return 0.125 * np.sin(t)**2 + (np.pi**2 / 6.0) * np.cos(t)**2

# ---------------------------------
# Quadrature: degree-2 (3 points)
# ---------------------------------
quad_bary = np.array([[2/3, 1/6, 1/6],
                      [1/6, 2/3, 1/6],
                      [1/6, 1/6, 2/3]])
quad_w = np.array([1/3, 1/3, 1/3])  # multiply by |T|

# ---------------------------------
# Mesh: uniform squares -> 2 tris
# ---------------------------------
def generate_mesh(nx, ny):
    hx, hy = 1.0/nx, 1.0/ny
    nodes = np.array([(i*hx, j*hy) for j in range(ny+1) for i in range(nx+1)], float)
    elems = []
    for j in range(ny):
        for i in range(nx):
            v0 = j*(nx+1) + i
            v1 = v0 + 1
            v2 = v0 + (nx+1)
            v3 = v2 + 1
            # two triangles per square
            elems.append([v0, v1, v3])
            elems.append([v0, v3, v2])
    return nodes, np.array(elems, dtype=int)

# ---------------------------------
# Triangle geom: area & grad φ_i
# ---------------------------------
def tri_area_and_grads(X):
    (x1,y1), (x2,y2), (x3,y3) = X
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    A = 0.5*abs(detJ)
    twoA = detJ  # keep orientation
    g1 = np.array([(y2 - y3), (x3 - x2)]) / twoA
    g2 = np.array([(y3 - y1), (x1 - x3)]) / twoA
    g3 = np.array([(y1 - y2), (x2 - x1)]) / twoA
    grads = np.vstack([g1, g2, g3])   # shape (3,2), constant on T
    return A, grads

# ---------------------------------
# Assembly: mass-lumped M_L and K
# ---------------------------------
def assemble_Mlumped_K(nodes, elems):
    """
    Assemble:
      - Lumped P1 mass matrix M_L (diagonal).
      - Stiffness matrix K with variable coefficient h(x,y),
        using the degree-2 (3-point) triangle quadrature.
    """
    N = len(nodes)
    Mdiag = np.zeros(N)
    I, J, V = [], [], []

    for tri in elems:
        idx = np.array(tri, int)
        X = nodes[idx]
        A, G = tri_area_and_grads(X)

        # Lumped mass: each node gets |T|/3
        # (equivalent to lumping the consistent P1 mass).
        Mdiag[idx] += A/3.0

        # Stiffness with 3-point quad (exact for quadratic h on a triangle)
        GGT = G @ G.T  # ∇φ_i · ∇φ_j matrix, constant on T
        Kloc = np.zeros((3,3))
        for q in range(3):
            lmb, mu, nu = quad_bary[q]
            xq = lmb*X[0,0] + mu*X[1,0] + nu*X[2,0]
            yq = lmb*X[0,1] + mu*X[1,1] + nu*X[2,1]
            wq = quad_w[q] * A
            hq = h_coef(xq, yq)
            Kloc += hq * GGT * wq

        for a in range(3):
            for b in range(3):
                I.append(idx[a]); J.append(idx[b]); V.append(Kloc[a,b])

    K = sp.coo_matrix((V,(I,J)), shape=(N,N)).tocsr()
    M = sp.diags(Mdiag, format='csr')
    return M, K

def assemble_load(nodes, elems, t):
    """
    Assemble load vector F(t) ≈ ∫ f(t,x,y) φ_i(x,y) dx,
    using the same degree-2 quadrature.
    """
    N = len(nodes)
    F = np.zeros(N)
    for tri in elems:
        idx = np.array(tri, int)
        X = nodes[idx]
        A, G = tri_area_and_grads(X)
        for q in range(3):
            lmb, mu, nu = quad_bary[q]
            xq = lmb*X[0,0] + mu*X[1,0] + nu*X[2,0]
            yq = lmb*X[0,1] + mu*X[1,1] + nu*X[2,1]
            wq = quad_w[q] * A
            fq = f(t, xq, yq)
            phi = np.array([lmb, mu, nu])
            F[idx] += fq * phi * wq
    return F

# ---------------------------------
# Dirichlet handling (homogeneous)
# ---------------------------------
def interior_boundary_sets(nodes, tol=1e-14):
    bnd = np.where((np.abs(nodes[:,0]) < tol) |
                   (np.abs(nodes[:,0]-1) < tol) |
                   (np.abs(nodes[:,1]) < tol) |
                   (np.abs(nodes[:,1]-1) < tol))[0]
    allidx = np.arange(len(nodes))
    intdofs = np.setdiff1d(allidx, bnd)
    return intdofs, bnd

# ---------------------------------
# Discrete energy: E_h = 1/2 (V^T M V + U^T K U)
# ---------------------------------
def fem_energy(Uvec, Vvec, M, K):
    kin = 0.5 * float(Vvec @ (M @ Vvec))
    pot = 0.5 * float(Uvec @ (K @ Uvec))
    return kin + pot

# ---------------------------------
# Explicit leapfrog (central difference) with lumped M
#   u^{n+1}_i = 2 u^n_i - u^{n-1}_i + dt^2 * (M_L^{-1})_ii ( F^n_i - (K u^n)_i )
# Taylor start (u_t(0)=0):
#   u^1 = u^0 + 0.5 dt^2 a^0,   a^0 = M_L^{-1}(F^0 - K u^0).
# ---------------------------------
def leapfrog_lumped(M, K, nodes, elems, dt, T, track_energy=True):
    intdofs, bnd = interior_boundary_sets(nodes)
    Ki = K[intdofs,:][:,intdofs]
    Minv_diag = 1.0 / M.diagonal()[intdofs]

    nsteps = int(round(T/dt))
    N = len(nodes)

    U = np.zeros((N, nsteps+1))
    V = np.zeros((N, nsteps+1))  # will fill after U is known (central diffs)

    # Initial conditions (from exact solution at t=0), u_t(0)=0
    U[:,0]   = exact_u(0.0, nodes[:,0], nodes[:,1])
    V[:,0]   = 0.0
    # Enforce Dirichlet at boundary nodes explicitly
    U[bnd,0] = 0.0
    V[bnd,0] = 0.0

    # Taylor start (n=1), consistent with u_t(0)=0
    F0 = assemble_load(nodes, elems, 0.0)[intdofs]
    a0 = Minv_diag * (F0 - Ki @ U[intdofs,0])
    U[intdofs,1] = U[intdofs,0] + 0.5*(dt*dt)*a0
    U[bnd,1] = 0.0

    # Main leapfrog loop
    for n in range(1, nsteps):
        tn = n*dt
        Fn = assemble_load(nodes, elems, tn)[intdofs]
        an = Minv_diag * (Fn - Ki @ U[intdofs,n])
        U[intdofs,n+1] = 2*U[intdofs,n] - U[intdofs,n-1] + (dt*dt)*an
        U[bnd,n+1] = 0.0

    # Build velocities by central differences for diagnostics
    for n in range(1, nsteps):
        V[intdofs,n] = (U[intdofs,n+1] - U[intdofs,n-1]) / (2*dt)
    # endpoints
    V[:,nsteps] = (U[:,nsteps] - U[:,nsteps-1]) / dt
    V[bnd,nsteps] = 0.0

    if track_energy:
        times = np.linspace(0.0, nsteps*dt, nsteps+1)
        E_num = [fem_energy(U[:,n], V[:,n], M, K) for n in range(nsteps+1)]
        E_ex  = [exact_energy(t) for t in times]
        return U, V, np.array(times), np.array(E_num), np.array(E_ex)
    else:
        return U, V, None, None, None

# ---------------------------------
# Errors via elementwise quadrature
# ---------------------------------
def errors_L2_H1(Uvec, nodes, elems, t, weighted_by_H=False):
    """
    Compute:
      - L2 error  ||u_h - u||_{L^2(Ω)}
      - H1 seminorm error |u_h - u|_{H^1(Ω)}
    If weighted_by_H=True, the H1 seminorm uses h(x,y)|∇e|^2,
    i.e. the energy seminorm induced by a(u,v)=∫ h ∇u·∇v.
    """
    eL2 = 0.0
    eH1 = 0.0
    for tri in elems:
        idx = np.array(tri, int)
        X = nodes[idx]
        A, G = tri_area_and_grads(X)

        # ∇u_h is constant per triangle
        uh_gx = np.sum(Uvec[idx] * G[:,0])
        uh_gy = np.sum(Uvec[idx] * G[:,1])

        for q in range(3):
            lmb, mu, nu = quad_bary[q]
            xq = lmb*X[0,0] + mu*X[1,0] + nu*X[2,0]
            yq = lmb*X[0,1] + mu*X[1,1] + nu*X[2,1]
            wq = quad_w[q]*A
            phi = np.array([lmb,mu,nu])
            uh = np.dot(phi, Uvec[idx])
            ue = exact_u(t, xq, yq)
            ex, ey = exact_grad_u(t, xq, yq)

            eL2 += (uh-ue)**2 * wq
            gx = uh_gx - ex
            gy = uh_gy - ey
            if weighted_by_H:
                eH1 += h_coef(xq,yq) * (gx*gx + gy*gy) * wq
            else:
                eH1 += (gx*gx + gy*gy) * wq
    return np.sqrt(eL2), np.sqrt(eH1)

# ---------------------------------
# Convergence driver
# ---------------------------------
def max_error_nodal(Uvec, nodes, t):
    ue = exact_u(t, nodes[:,0], nodes[:,1])
    return np.max(np.abs(Uvec - ue))

def run_once(nx, ny, T, c_dt=0.25, track_energy=False):
    """
    Run FEM+leapfrog on an nx×ny uniform triangulation.
    Time step:
        dt ≈ c_dt * h_mesh / sqrt(h_max),
    where h_max = max h(x,y) on Ω = [0,1]^2.
    For h(x,y)=x^2+y^2, we have h_max = 2.
    """
    nodes, elems = generate_mesh(nx, ny)
    h_mesh = 1.0 / nx
    h_max = 2.0  # max_{Ω} h(x,y) for h=x^2+y^2 on [0,1]^2
    dt = c_dt * h_mesh / np.sqrt(h_max)

    M, K = assemble_Mlumped_K(nodes, elems)

    # Explicit mass-lumped leapfrog
    U, V, times, E_num, E_ex = leapfrog_lumped(M, K, nodes, elems, dt, T, track_energy=track_energy)

    # errors at final time
    eL2, eH1 = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=False)
    eL2H, eH1H = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=True)
    eMax = max_error_nodal(U[:, -1], nodes, T)

    return h_mesh, dt, eL2, eH1, eL2H, eH1H, eMax, (nodes, elems, U, V, times, E_num, E_ex)

def plot_mesh(nodes, elems, nx=None, ny=None, zoom_cells=5, tol=1e-12):
    """
    Plot full triangulation + a zoomed view near (0,0),
    highlighting interior (red) and boundary (blue) nodes.
    """
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
    intdofs, bnd = interior_boundary_sets(nodes, tol=tol)

    # choose zoom window size based on how many cells to show (uniform [0,1]^2 grid)
    if nx is None or ny is None:
        zx = zy = 0.2
    else:
        zx = zoom_cells / float(nx)   # width of zoom window
        zy = zoom_cells / float(ny)   # height of zoom window

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Full mesh
    ax = axes[0]
    ax.triplot(tri, lw=0.6,  alpha=0.5, color='k')
    ax.scatter(nodes[intdofs, 0], nodes[intdofs, 1], s=10, c='tab:red',
               edgecolors='none', zorder=3, alpha=0.9, label='Interior nodes')
    ax.scatter(nodes[bnd, 0], nodes[bnd, 1], s=15, c='b',
               edgecolors='none', zorder=3, label='Boundary nodes')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0)
    ttl = 'Triangulation mesh'
    if nx is not None and ny is not None:
        ttl += f' (nx={nx}, ny={ny})'
    ax.set_title(ttl)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', frameon=True)

    # highlight the zoom region on the full mesh
    rect = Rectangle((0.0, 0.0), zx, zy, fill=False, ec='red', lw=1.2)
    ax.add_patch(rect)

    # --- Zoomed mesh (bottom-left corner)
    axz = axes[1]
    axz.triplot(tri, lw=0.6, alpha=0.5, color='k')

    # boundary nodes inside the zoom window
    mask_b = (nodes[bnd, 0] <= zx + tol) & (nodes[bnd, 1] <= zy + tol)
    bnd_zoom = bnd[mask_b]
    mask_i = (nodes[intdofs, 0] <= zx + tol) & (nodes[intdofs, 1] <= zy + tol)
    int_zoom = intdofs[mask_i]
    axz.scatter(nodes[int_zoom, 0], nodes[int_zoom, 1], s=20, c='r',
                edgecolors='none', zorder=3)
    axz.scatter(nodes[bnd_zoom, 0], nodes[bnd_zoom, 1], s=20, c='b',
                edgecolors='none', zorder=3)

    axz.set_aspect('equal', adjustable='box')
    axz.set_xlim(0.0, zx); axz.set_ylim(0.0, zy)
    axz.set_title(f'Zoomed-in mesh (bottom-left {zx:.3f}×{zy:.3f})')
    axz.set_xlabel('x'); axz.set_ylabel('y')
    axz.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    Tfinal = 1.0

    # One representative run with energy tracking + plot
    nx_plot = ny_plot = 20
    h_mesh, dt, *_errs, pack = run_once(nx_plot, ny_plot, Tfinal, c_dt=0.25, track_energy=True)
    nodes, elems, U, V, times, E_num, E_ex = pack

    # Energy comparison plot
    plt.figure(figsize=(8,5))
    plt.plot(times, E_num,  'b-', label='Numerical energy $E_h(t)$')
    plt.plot(times, E_ex, 'r--', label='Exact energy $E_{\\mathrm{exact}}(t)$')
    plt.xlabel('t'); plt.ylabel('Energy')
    plt.title(f'Energy (FEM leapfrog vs Exact), nx=ny={nx_plot}, dt={dt:.4e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Build structured (ny+1, nx+1) grids from the uniform mesh
    x = np.linspace(0.0, 1.0, nx_plot + 1)
    y = np.linspace(0.0, 1.0, ny_plot + 1)
    X, Y = np.meshgrid(x, y)  # shapes (ny+1, nx+1)

    u_num_grid = U[:, -1].reshape((ny_plot + 1, nx_plot + 1))
    u_exact_T = exact_u(Tfinal, X, Y)
    err_grid = np.abs(u_num_grid - u_exact_T)

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, u_num_grid, cmap='viridis', rstride=1, cstride=1)
    ax1.set_title('Numerical Solution')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_exact_T, cmap='viridis', rstride=1, cstride=1)
    ax2.set_title('Exact Solution')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, err_grid, cmap='inferno', rstride=1, cstride=1)
    ax3.set_title('Error')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('abs error')

    plt.tight_layout()
    plt.show()

    # Visualize triangulation + zoomed corner
    plot_mesh(nodes, elems, nx=nx_plot, ny=ny_plot, zoom_cells=5)

    # Optional: report final relative energy error
    relE = abs(E_num[-1] - E_ex[-1]) / max(1e-14, E_ex[-1])
    print(f"Final energy: FEM={E_num[-1]:.8e}, exact={E_ex[-1]:.8e}, rel. error={relE:.3e}")

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")

    tracemalloc.stop()

