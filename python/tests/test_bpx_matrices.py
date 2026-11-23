import pytest
import numpy as np
import hmod.polynomial_bases as pb
from hmod.preconditioning import BPXPreconditioner
from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives

def _test_lagrange_prolongations(polynomial_degree: int):
    levels = 3
    nt_coarse = 3
    T = 1.5
    fun = lambda t : np.cos(t)
    points_coarse = pb.get_langrange_points(polynomial_degree, nt_coarse, T)
    dof_vals_coarse = fun(points_coarse)
    trans = pb.get_lagrange_to_legendre_matrix(polynomial_degree, nt_coarse)
    evaluator = pb.LegendreBasisEvaluator(trans@dof_vals_coarse, polynomial_degree, nt_coarse, T)
    t_check = np.linspace(0, T, 105)
    vals_expected = evaluator.evaluate(t_check)
    for level in range(levels):
        nt_fine = nt_coarse * (2**level)
        P = pb.get_lagrange_prolongation_matrix(nt_coarse, nt_fine, polynomial_degree)
        dof_vals_fine = P @ dof_vals_coarse
        trans_fine = pb.get_lagrange_to_legendre_matrix(polynomial_degree, nt_fine)
        evaluator_fine = pb.LegendreBasisEvaluator(trans_fine @ dof_vals_fine, polynomial_degree, nt_fine, T)
        vals_fine = evaluator_fine.evaluate(t_check)
        diff = np.abs(vals_expected - vals_fine)
        assert np.max(diff) < 1e-10


def test_lagrange_prolongations():
    for polynomial_degree in [1, 2, 3, 4, 5]:
        _test_lagrange_prolongations(polynomial_degree)


def test_bpx_weight_on_coarse_linear_function():
    T = 1.0
    nt_coarse = 1
    n_refinements = 2
    nt_fine = nt_coarse * 2**n_refinements
    preconditioner = BPXPreconditioner(
        mu=100, n_refinements=n_refinements, sobolev_exponent=0.5,
        polynomial_degree=1, nt_coarse=nt_coarse, T=T, periodic=False,
    )
    M_fine = get_lagrange_lagrange_matrix_for_derivatives(
        1, 1, 0, 0, nt_fine, T, periodic=False,
    )[1:, 1:]
    # Active nodal coefficients of v(t) = t with the initial value removed.
    v = np.linspace(0.0, T, nt_fine + 1)[1:]
    np.testing.assert_allclose(preconditioner @ (M_fine @ v), v / 101)


if __name__ == "__main__":
    pytest.main([__file__])
