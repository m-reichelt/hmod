import pytest

from hmod.preconditioning import BPXPreconditioner
import hmod.hilbert_matrices as hm
import hmod.standard_matrices as sm
from hmod.dof_handling import DofRestrictorSymmetric
import numpy as np


def _test_bpx_only_time_derivative_for_settings(polynomial_degree: int, n_coarsest: int, n_refinements: int):
    T = 1.0
    mu = 0.
    nmodes = int(1e5)
    n_finest = n_coarsest * (2 ** n_refinements)
    bpx = BPXPreconditioner(mu, n_refinements, 0.5, polynomial_degree, n_coarsest, T)
    At = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(n_finest, polynomial_degree, polynomial_degree, 1, 0,
                                                                  T)

    # bpx_0 = DofRestrictorSymmetric(bpx, np.array([0]), np.array([0.]))
    bpx_0 = bpx  # already includes BC
    At_0 = DofRestrictorSymmetric(At, np.array([0]), np.array([0.]))

    # get rhs vector
    ndofs = At_0.shape[0]
    rhs0 = np.ones(ndofs)

    # solve using gmres with bpx preconditioner
    from scipy.sparse.linalg import gmres
    rtol = 1e-10
    maxiter = ndofs + 1
    gmres_iter = 0

    def callback(rk):
        nonlocal gmres_iter
        gmres_iter += 1

    solution_bpx, info_bpx = gmres(At_0, rhs0, M=bpx_0, rtol=rtol, maxiter=maxiter, restart=maxiter, callback=callback)
    assert info_bpx == 0, f"GMRES with BPX preconditioner did not converge, info: {info_bpx}"
    iter_bpx = gmres_iter

    print(
        f"GMRES with BPX preconditioner converged in {iter_bpx} iterations for polynomial degree {polynomial_degree}, n_coarsest {n_coarsest}, n_refinements {n_refinements}")

    return iter_bpx


def _test_bpx_hybrid_for_settings(mu: float, polynomial_degree: int, n_coarsest: int, n_refinements: int):
    T = 1.0
    nmodes = int(1e5)
    n_finest = n_coarsest * (2 ** n_refinements)
    import hmod.deprecated_hilbert_matrices as hmd
    AtH = hmd.Operator_dt_H_Lagrange_Lagrange(nmodes, n_finest, polynomial_degree, polynomial_degree)
    MtH = hmd.Operator_I_H_Lagrange_Lagrange(nmodes, n_finest, polynomial_degree, polynomial_degree)
    AtH = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(n_finest, polynomial_degree, polynomial_degree, 1, 0,
                                                                  T)
    KH = AtH + mu * MtH
    AtS = sm.get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree, polynomial_degree, 1, 0, n_finest, T)
    MtS = sm.get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree, polynomial_degree, 0, 0, n_finest, T)
    KS = AtS + mu * MtS
    # transform KS to linear operator
    from scipy.sparse.linalg import LinearOperator
    KS_op = LinearOperator(dtype=np.float64, shape=KS.shape, matvec=lambda x: KS @ x)

    K = KH + KS_op

    bpx = BPXPreconditioner(mu, n_refinements, 0.5, polynomial_degree, n_coarsest, T)
    bpx_0 = DofRestrictorSymmetric(bpx, np.array([0]), np.array([0.]))
    bpx_0 = bpx  # already includes BC
    K_0 = DofRestrictorSymmetric(K, np.array([0]), np.array([0.]))

    # get rhs vector
    ndofs = K_0.shape[0]
    rhs0 = np.ones(ndofs)

    # solve using gmres with bpx preconditioner
    from scipy.sparse.linalg import gmres
    rtol = 1e-10
    maxiter = 150
    gmres_iter = 0

    def callback(rk):
        nonlocal gmres_iter
        gmres_iter += 1

    solution_bpx, info_bpx = gmres(K_0, rhs0, M=bpx_0, rtol=rtol, maxiter=maxiter, restart=maxiter, callback=callback)
    assert info_bpx == 0, f"GMRES with BPX preconditioner did not converge, info: {info_bpx}"
    iter_bpx = gmres_iter

    print(
        f"GMRES with BPX preconditioner converged in {iter_bpx} iterations for polynomial degree {polynomial_degree}, n_coarsest {n_coarsest}, n_refinements {n_refinements}")

    return iter_bpx


def _test_bpx_preconditioner_only_time_derivative():
    n_coarsest = 4
    n_refinements = 4
    degrees_to_test = [1, 2, 3, 4, 5]
    gmres_limit = [20, 30, 40, 50, 60]
    for pol_degree, limit in zip(degrees_to_test, gmres_limit):
        gmres_iter = _test_bpx_only_time_derivative_for_settings(pol_degree, n_coarsest, n_refinements)
        assert gmres_iter < limit, f"GMRES with BPX preconditioner took too many iterations: {gmres_iter} for polynomial degree {pol_degree}"


def test_bpx_hybrid():
    n_coarsest = 4
    n_refinements = 4
    degrees_to_test = [1, 2, 3, 4, 5]
    gmres_limit = [60, 70, 90, 110, 120]
    mus = [0., 1., 10., 100.]
    for mu in mus:
        for pol_degree, limit in zip(degrees_to_test, gmres_limit):
            gmres_iter = _test_bpx_hybrid_for_settings(mu, pol_degree, n_coarsest, n_refinements)
            assert gmres_iter < limit, f"GMRES with BPX preconditioner took too many iterations: {gmres_iter} for polynomial degree {pol_degree}"


def _test_bpx_hybrid_single():
    n_coarsest = 4
    n_refinements = 8
    mu = 100.
    pol_degree = 2
    gmres_iter = _test_bpx_hybrid_for_settings(mu, pol_degree, n_coarsest, n_refinements)
    gmres_iter = _test_bpx_only_time_derivative_for_settings(pol_degree, n_coarsest, n_refinements)


if __name__ == "__main__":
    pytest.main([__file__])
