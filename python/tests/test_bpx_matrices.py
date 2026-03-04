import pytest
import numpy as np
import hmod.polynomial_bases as pb

def test_legendre_prolongations():
    levels = 3
    polynomial_degree = 1
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

    a = 1


if __name__ == "__main__":
    pytest.main([__file__])