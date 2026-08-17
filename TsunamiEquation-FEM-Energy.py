import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import timeit
import tracemalloc

tracemalloc.start()  # start tracing

# -------------------------------
# Exact solution, forcing, coeff
# -------------------------------
start = timeit.default_timer()

def exact_u(t, x, y):
    return np.cos(t) * np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_grad_u(t, x, y):
    ux = np.cos(t) * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uy = np.cos(t) * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return ux, uy

def f(t, x, y):
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    term_g = 2 * np.pi * (x * np.sin(np.pi * y) * np.cos(np.pi * x)
                          + y * np.sin(np.pi * x) * np.cos(np.pi * y))
    term_l = -2 * (np.pi ** 2) * (1 + x ** 2 + y ** 2) * S
    return -np.cos(t) * (S + term_g + term_l)

def H(x, y):
    return 1+ x**2 + y**2

# ---- Exact energy for comparison: E_exact(t) = 1/8 sin^2 t + (π^2/6) cos^2 t
def exact_energy(t):
    return 0.125 * np.sin(t)**2 + (5* np.pi**2 / 12.0) * np.cos(t)**2

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
# ∇φ1 = [(y2 - y3), (x3 - x2)]/(2A), etc.
# ---------------------------------
def tri_area_and_grads(X):
    (x1,y1), (x2,y2), (x3,y3) = X
    detJ = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    A = 0.5*abs(detJ)
    twoA = detJ  # keep orientation
    g1 = np.array([(y2 - y3), (x3 - x2)]) / twoA
    g2 = np.array([(y3 - y1), (x1 - x3)]) / twoA
    g3 = np.array([(y1 - y2), (x2 - x1)]) / twoA
    grads = np.vstack([g1, g2, g3])   # shape (3,2)
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

        # Stiffness with 3-pt quad (exact for H quadratic)
        Kloc = np.zeros((3,3))
        for q in range(3):
            lmb, mu, nu = quad_bary[q]
            xq = lmb*X[0,0] + mu*X[1,0] + nu*X[2,0]
            yq = lmb*X[0,1] + mu*X[1,1] + nu*X[2,1]
            wq = quad_w[q] * A
            Hq = H(xq, yq)
            Kloc += Hq * (G @ G.T) * wq  # (3x2)(2x3)->(3x3)

        for a in range(3):
            for b in range(3):
                I.append(idx[a]); J.append(idx[b]); V.append(Kloc[a,b])

    K = sp.coo_matrix((V,(I,J)), shape=(N,N)).tocsr()
    M = sp.diags(Mdiag, format='csr')
    return M, K

def assemble_load(nodes, elems, t):
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
# Discrete energy: E_h = 1/2 (V^T M V + U^T K U)
# ---------------------------------
def fem_energy(Uvec, Vvec, M, K):
    # works with lumped diagonal M and sparse K
    kin = 0.5 * float(Vvec @ (M @ Vvec))
    pot = 0.5 * float(Uvec @ (K @ Uvec))
    return kin + pot

# ---------------------------------
# Newmark (β=1/4, γ=1/2) with trapezoidal forcing
#    Returns U,V,A plus (times, E_num, E_ex)
# ---------------------------------
def newmark_beta14(M, K, nodes, elems, dt, T, track_energy=True):
    intdofs, _ = interior_boundary_sets(nodes)
    Mi = M[intdofs,:][:,intdofs]
    Ki = K[intdofs,:][:,intdofs]

    nsteps = int(round(T/dt))
    N = len(nodes)
    U = np.zeros((N, nsteps+1))
    V = np.zeros((N, nsteps+1))
    A = np.zeros((N, nsteps+1))

    # Initial conditions (interpolation)
    U[:,0] = exact_u(0.0, nodes[:,0], nodes[:,1])
    V[:,0] = 0.0

    # A^0: M A^0 = F(0) - K U^0 (interior)
    F0 = assemble_load(nodes, elems, 0.0)[intdofs]
    A[intdofs,0] = spla.spsolve(Mi, F0 - Ki @ U[intdofs,0])

    beta, gamma = 0.25, 0.5
    Keff = Ki + (1.0/(beta*dt*dt)) * Mi      # SPD, constant
    solve = spla.factorized(Keff.tocsc())

    times = []
    E_num = []
    E_ex  = []

    if track_energy:
        times.append(0.0)
        E_num.append(fem_energy(U[:,0], V[:,0], M, K))
        E_ex.append(exact_energy(0.0))

    Fprev = F0
    for n in range(nsteps):
        tn1 = (n+1)*dt
        Fnp1 = assemble_load(nodes, elems, tn1)[intdofs]
        # trapezoidal forcing average
        Fbar = 0.5*(Fnp1 + Fprev)

        # predictor
        Uhat = U[intdofs,n] + dt*V[intdofs,n] + 0.5*dt*dt*(1-2*beta)*A[intdofs,n]

        # solve for U^{n+1}
        RHS = Fbar + (1.0/(beta*dt*dt)) * (Mi @ Uhat)
        Ui1 = solve(RHS)

        # update A and V
        Ai1 = (1.0/(beta*dt*dt)) * (Ui1 - Uhat)
        Vi1 = V[intdofs,n] + dt*((1-gamma)*A[intdofs,n] + gamma*Ai1)

        U[intdofs,n+1] = Ui1
        V[intdofs,n+1] = Vi1
        A[intdofs,n+1] = Ai1

        if track_energy:
            times.append(tn1)
            E_num.append(fem_energy(U[:,n+1], V[:,n+1], M, K))
            E_ex.append(exact_energy(tn1))

        Fprev = Fnp1

    if track_energy:
        return U, V, A, np.array(times), np.array(E_num), np.array(E_ex)
    else:
        return U, V, A, None, None, None

# ---------------------------------
# Errors via elementwise quadrature
# ---------------------------------
def errors_L2_H1(Uvec, nodes, elems, t, weighted_by_H=False):
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
                eH1 += H(xq,yq) * (gx*gx + gy*gy) * wq
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
    nodes, elems = generate_mesh(nx, ny)
    h = 1.0 / nx
    dt = c_dt * h
    M, K = assemble_Mlumped_K(nodes, elems)

    U, V, A, times, E_num, E_ex = newmark_beta14(M, K, nodes, elems, dt, T, track_energy=track_energy)

    # errors at final time
    eL2, eH1 = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=False)
    eL2H, eH1H = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=True)
    eMax = max_error_nodal(U[:, -1], nodes, T)

    return h, dt, eL2, eH1, eL2H, eH1H, eMax, (nodes, elems, U, V, times, E_num, E_ex)

if __name__ == "__main__":
    Tfinal = 1.0


    # ny = nx = 20
    # h, dt, eL2, eH1, eL2H, eH1H, eMax, _ = run_once(nx, ny, Tfinal, c_dt=0.25, track_energy=False)
    # print(f"nx=ny={nx:3d}, h={h:.5f}, dt={dt:.5f}  |  "f"Max={eMax:.3e},  L2={eL2:.3e},  H1={eH1:.3e},  H1(H)={eH1H:.3e}")

    # One representative run with energy tracking + plot
    nx_plot = ny_plot = 20
    h, dt, *_errs, pack = run_once(nx_plot, ny_plot, Tfinal, c_dt=0.25, track_energy=True)
    nodes, elems, U, V, times, E_num, E_ex = pack

    # Energy comparison plot
    plt.figure(figsize=(8,5))
    plt.plot(times, E_num,  'b-', fillstyle='none',  markersize=4, label='Numerical energy $E_h(t)$')
    plt.plot(times, E_ex, 'ro', fillstyle='none',  markersize=4, label='Exact energy $E_{\\mathrm{exact}}(t)$')
    plt.xlabel('t'); plt.ylabel('Energy')
    plt.title(f'Energy (FEM vs Exact), nx=ny={nx_plot}, dt={dt:.4e}')
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
    ax1.set_title('Numerical Solution at T')
    ax1.set_xlabel('x');
    ax1.set_ylabel('y');
    ax1.set_zlabel('u')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, u_exact_T, cmap='viridis', rstride=1, cstride=1)
    ax2.set_title('Exact Solution at T')
    ax2.set_xlabel('x');
    ax2.set_ylabel('y');
    ax2.set_zlabel('u')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, err_grid, cmap='inferno', rstride=1, cstride=1)
    ax3.set_title('Error |u_num - u_exact|')
    ax3.set_xlabel('x');
    ax3.set_ylabel('y');
    ax3.set_zlabel('abs error')

    plt.tight_layout()
    plt.show()

    # Optional: report final relative energy error
    relE = abs(E_num[-1] - E_ex[-1]) / max(1e-14, E_ex[-1])
    print(f"Final energy: FEM={E_num[-1]:.8e}, exact={E_ex[-1]:.8e}, rel. error={relE:.3e}")
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 ** 2:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 ** 2:.2f} MB")

    tracemalloc.stop()