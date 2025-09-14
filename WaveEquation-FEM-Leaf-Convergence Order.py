import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla  # (unused now, ok to remove)

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
    S = np.sin(np.pi * x) * np.sin(np.pi * y)
    term_g = 2 * np.pi * (x * np.sin(np.pi * y) * np.cos(np.pi * x)
                          + y * np.sin(np.pi * x) * np.cos(np.pi * y))
    term_l = -2 * (np.pi ** 2) * (x ** 2 + y ** 2) * S
    return -np.cos(t) * (S + term_g + term_l)

def H(x, y):
    return x**2 + y**2

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
# Explicit leapfrog (central difference) with lumped M
# M u'' + K u = F(t)  =>  u^{n+1} = 2u^n - u^{n-1} + dt^2 M^{-1}(F^n - K u^n)
# Taylor start: u^1 = u^0 + dt v^0 + 0.5 dt^2 a^0,   a^0 = M^{-1}(F^0 - K u^0)
# ---------------------------------
def leapfrog_lumped(M, K, nodes, elems, dt, T):
    intdofs, bnd = interior_boundary_sets(nodes)
    Ki = K[intdofs,:][:,intdofs]
    Minv_diag = 1.0 / M.diagonal()[intdofs]

    nsteps = int(round(T/dt))
    N = len(nodes)
    U = np.zeros((N, nsteps+1))

    # Initial displacement from exact solution; v0 = 0; enforce Dirichlet on boundary
    U[:,0]   = exact_u(0.0, nodes[:,0], nodes[:,1])
    U[bnd,0] = 0.0

    # Start step (n=1)
    F0 = assemble_load(nodes, elems, 0.0)[intdofs]
    a0 = Minv_diag * (F0 - Ki @ U[intdofs,0])
    U[intdofs,1] = U[intdofs,0] + 0.5*(dt*dt)*a0
    U[bnd,1] = 0.0

    # Main loop
    for n in range(1, nsteps):
        tn = n*dt
        Fn = assemble_load(nodes, elems, tn)[intdofs]
        an = Minv_diag * (Fn - Ki @ U[intdofs,n])
        U[intdofs,n+1] = 2*U[intdofs,n] - U[intdofs,n-1] + (dt*dt)*an
        U[bnd,n+1] = 0.0

    return U

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
    """Max nodal (infinity-norm) error at time t."""
    ue = exact_u(t, nodes[:,0], nodes[:,1])
    return np.max(np.abs(Uvec - ue))

def run_once(nx, ny, T, c_dt=0.25):
    nodes, elems = generate_mesh(nx, ny)
    h = 1.0 / nx
    dt = c_dt * h                     # CFL-like choice
    M, K = assemble_Mlumped_K(nodes, elems)
    U = leapfrog_lumped(M, K, nodes, elems, dt, T)  # <<< explicit leapfrog

    # errors at final time
    eL2, eH1 = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=False)
    eL2H, eH1H = errors_L2_H1(U[:, -1], nodes, elems, T, weighted_by_H=True)
    eMax = max_error_nodal(U[:, -1], nodes, T)
    return h, dt, eL2, eH1, eL2H, eH1H, eMax

if __name__ == "__main__":
    Tfinal = 1.0
    for nx in [20, 40, 80, 160]:
        ny = nx
        h, dt, eL2, eH1, eL2H, eH1H, eMax = run_once(nx, ny, Tfinal, c_dt=0.25)
        print(f"nx=ny={nx:3d}, h={h:.5f}, dt={dt:.5f}  |  "
              f"Max={eMax:.3e},  L2={eL2:.3e},  H1={eH1:.3e},  H1(H)={eH1H:.3e}")
