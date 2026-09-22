"""Hybrid ODE convergence study; requires an installed hmodFFT package.

Example: python skripts/ode_convergence.py --mu 3 --n-refinements 6 --periodic
The H12 term uses the spectral (modified/periodic Hilbert) norm from hmod.
"""

import argparse
import csv

import numpy as np
from scipy.sparse.linalg import aslinearoperator, gmres

import hmod.hilbert_matrices as hm
import hmod.polynomial_bases as pb
import hmod.standard_matrices as sm
from hmod.dof_handling import DofRestrictorSymmetric
from hmod.norms import compute_h12_seminorm, compute_l2_norm
from hmod.preconditioning import BPXPreconditioner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mu", type=float, default=3.0)
parser.add_argument("--n-refinements", type=int, default=8)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--periodic", action="store_true", help="Use u(0)=u(T) and T=2.")
parser.add_argument("--output", default="convergence.csv")
args = parser.parse_args()
mu, p, periodic = args.mu, args.degree, args.periodic
if args.n_refinements < 0 or p < 1 or mu < 0:
    parser.error("Require n-refinements >= 0, degree >= 1 and mu >= 0.")
if periodic and mu == 0:
    parser.error("Require mu > 0 for a unique periodic solution.")

print(f"Periodic={periodic}, mu={mu}, degree={p}, n_refinements={args.n_refinements}")

T = 2.0
u = (lambda t: np.cos(np.pi * t)) if periodic else (lambda t: np.sin(np.pi * t))
du = (lambda t: -np.pi * np.sin(np.pi * t)) if periodic else (
    lambda t: np.pi * np.cos(np.pi * t)
)
f = lambda t: du(t) + mu * u(t)
rows = []
previous = None

for level in range(args.n_refinements + 1):
    n = 2 * 2**level
    # Test with v + H(v), using continuous piecewise polynomials of degree p.
    A = sm.get_lagrange_lagrange_matrix_for_derivatives(
        p, p, 1, 0, n, T, periodic=periodic)
    M = sm.get_lagrange_lagrange_matrix_for_derivatives(
        p, p, 0, 0, n, T, periodic=periodic)
    AH = hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(
        p, p, 1, 0, n, T, periodic=periodic)
    MH = hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(
        p, p, 0, 0, n, T, periodic=periodic)
    B = AH + aslinearoperator(A) + mu * (MH + aslinearoperator(M))

    F = sm.get_legendre_lagrange_matrix_for_derivatives(
        p, p, 0, 0, n, T, periodic=periodic)
    FH = hm.get_hilbert_matrix_for_derivatives_legendre_lagrange(
        p, p, 0, 0, n, T, periodic=periodic)
    fh = sm.project_rhs_onto_legendre_basis(f, n, p, T, periodic=periodic)
    rhs = (aslinearoperator(F) + FH) @ fh
    if not periodic:
        B = DofRestrictorSymmetric(B, [0], [0.0])
        rhs = rhs[1:]

    preconditioner = BPXPreconditioner(mu, level, 0.5, p, 2, T, periodic=periodic)
    uh, info = gmres(B, rhs, M=preconditioner, rtol=1e-12)
    if info != 0:
        raise RuntimeError(f"GMRES failed for n={n}: info={info}")
    if not periodic:
        uh = np.r_[0.0, uh]

    transform = pb.get_lagrange_to_legendre_matrix(p, n, periodic=periodic)
    solution = pb.LegendreBasisEvaluator(transform @ uh, p, n, T)
    error = lambda t: u(t) - solution.evaluate(t)
    l2 = compute_l2_norm(error, n, T)
    # Keep at least 32 samples per element in the spectral norm calculation.
    h12 = compute_h12_seminorm(
        error, T, quad_tol=min(1e-8, (16.0 * n)**-2), periodic=periodic)
    energy = np.sqrt(h12**2 + mu * l2**2)
    errors = np.array([l2, energy])
    eoc = np.log2(previous / errors) if previous is not None else [np.nan, np.nan]
    rows.append([n, T / n, mu, p, T, l2, eoc[0], energy, eoc[1]])
    previous = errors
    print(f"n={n:5d}  L2={l2:.6e}  X={energy:.6e}")

with open(args.output, "w", newline="", encoding="ascii") as output:
    writer = csv.writer(output)
    writer.writerow(["n", "h", "mu", "p", "T", "L2error", "L2eoc", "Xerror", "Xeoc"])
    writer.writerows(rows)
print(f"Saved {args.output}")
