import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import timeit
import tracemalloc

tracemalloc.start()
start = timeit.default_timer()

# -------------------------------
# Case selector for h(x)
# -------------------------------
SELECT_CASE = 1  # change to 1 or 2

def h_case_vals(x, case_id):
    x = np.asarray(x)
    if case_id == 1:
        h = np.where(x < 0.75, 1.0, 0.1)
    elif case_id == 2:
        h = np.where(x < 0.3, 1.0,
                     np.where(x < 0.8, 1.0 - 1.8*(x - 0.3), 0.1))
    else:
        raise ValueError("Unknown case_id (use 1 or 2).")
    return h

def H(x, y):
    # Coefficient depends on x only via the selected case
    return h_case_vals(x, SELECT_CASE)

# ---------------------------------
# Initial condition (Gaussian bump)
# ---------------------------------
def gaussian_initial(Xc, Yc):
    return 2*np.exp(-((Xc-0.5)**2)/0.002 - ((Yc-0.5)**2)/0.002)

# ---------------------------------
# Quadrature: degree-2 (3 point)
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
    twoA = detJ  # orientation
    g1 = np.array([(y2 - y3), (x3 - x2)]) / twoA
    g2 = np.array([(y3 - y1), (x1 - x3)]) / twoA
    g3 = np.array([(y1 - y2), (x2 - x1)]) / twoA
    grads = np.vstack([g1, g2, g3])   # (3,2)
    return A, grads

# ---------------------------------
# Assembly (COO -> CSR)
# ---------------------------------
def assemble_Mlumped_K(nodes, elems):
    N = len(nodes)
    Mdiag = np.zeros(N)
    I, J, V = [], [], []

    for tri in elems:
        idx = np.array(tri, int)
        X = nodes[idx]
        A, G = tri_area_and_grads(X)

        # Lumped mass: each node gets |T|/3
        Mdiag[idx] += A/3.0

        # Stiffness with 3-pt quad, H(xq)
        Kloc = np.zeros((3,3))
        for q in range(3):
            lmb, mu, nu = quad_bary[q]
            xq = lmb*X[0,0] + mu*X[1,0] + nu*X[2,0]
            yq = lmb*X[0,1] + mu*X[1,1] + nu*X[2,1]
            wq = quad_w[q] * A
            Hq = H(xq, yq)  # depends on x only
            Kloc += Hq * (G @ G.T) * wq  # (3x2)(2x3)->(3x3)

        for a in range(3):
            for b in range(3):
                I.append(idx[a]); J.append(idx[b]); V.append(Kloc[a,b])

    K = sp.coo_matrix((V,(I,J)), shape=(N,N)).tocsr()
    M = sp.diags(Mdiag, format='csr')
    return M, K

# ---------------------------------
# Dirichlet handling
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
# Explicit central-difference (leapfrog) with lumped M
# u^{n+1}_i = 2 u^n_i - u^{n-1}_i + dt^2 * M^{-1}(-K u^n)_i
# ---------------------------------
def fem_central_diff_lumped(M, K, nodes, dt, T):
    intdofs, bnd = interior_boundary_sets(nodes)
    Ki = K[intdofs,:][:,intdofs]              # interior block
    Minv_diag = 1.0 / M.diagonal()[intdofs]   # lumped mass inverse (vector)

    nsteps = int(round(T/dt))
    N = len(nodes)
    U = np.zeros((N, nsteps+1))

    # Initial displacement (Gaussian), zero velocity; enforce Dirichlet on boundary
    U[:,0]   = gaussian_initial(nodes[:,0], nodes[:,1])
    U[bnd,0] = 0.0

    # Taylor warm start: u^1 = u^0 + 0.5 dt^2 * a^0,  a^0 = M^{-1}(-K u^0)
    a0 = -(Ki @ U[intdofs,0])
    U[intdofs,1] = U[intdofs,0] + 0.5*(dt*dt)*(Minv_diag * a0)
    U[bnd,1] = 0.0

    # Time stepping
    for n in range(1, nsteps):
        an = -(Ki @ U[intdofs,n])
        U[intdofs,n+1] = 2*U[intdofs,n] - U[intdofs,n-1] + (dt*dt)*(Minv_diag * an)
        U[bnd,n+1] = 0.0

    return U

# ---------------------------------
def run_once(nx, ny, T, c_dt=0.25):
    nodes, elems = generate_mesh(nx, ny)
    h = 1.0 / nx
    dt = c_dt * h                     # matches your FDM-style dt
    M, K = assemble_Mlumped_K(nodes, elems)
    U = fem_central_diff_lumped(M, K, nodes, dt, T)
    return h, dt, (nodes, elems, U)

# ---------------------------------
# Main
# ---------------------------------
if __name__ == "__main__":
    Tfinal = 0.75
    nx_plot = ny_plot = 160

    h, dt, pack = run_once(nx_plot, ny_plot, Tfinal, c_dt=0.25)
    nodes, elems, U = pack

    # Final-time surface (numerical u at T)
    x = np.linspace(0.0, 1.0, nx_plot + 1)
    y = np.linspace(0.0, 1.0, ny_plot + 1)
    X, Y = np.meshgrid(x, y)
    u_num_grid = U[:, -1].reshape((ny_plot + 1, nx_plot + 1))

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.subplots(subplot_kw={"projection": "3d"})
    surf = ax1.plot_surface(X, Y, u_num_grid, cmap='viridis', edgecolor='black',
                            linewidth=0.2, rstride=1, cstride=1)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_title(f'FEM (central diff, lumped M) at T={Tfinal} (case={SELECT_CASE})')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
    ax1.view_init(elev=30, azim=110)
    plt.tight_layout()
    plt.show()

    print("dx=", dt, "maximum U :", np.max(u_num_grid))
    # report timing + memory
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")
    tracemalloc.stop()
