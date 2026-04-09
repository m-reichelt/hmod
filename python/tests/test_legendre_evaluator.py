import pytest
import numpy as np
import hmod.polynomial_bases as pb

def test_evaluation_and_derivative():
    from hmod.polynomial_bases import get_langrange_points, get_lagrange_to_legendre_matrix, get_legendre_derivative_matrix
    import numpy as np
    nt = 40
    T = 5.0
    max_degree= 4
    pol_degrees_trial = [k+1 for k in range(0,max_degree)]
    for ptrial in pol_degrees_trial:
        #define a polynomial of of degree ptrial, so the derivative must be exact
        coeffs = [1.0 for _ in range(ptrial+1)]
        u_analytical = np.poly1d(coeffs)
        dt_u_analytical = u_analytical.deriv()
        #define a polynomial function
        lagrange_points = get_langrange_points(ptrial, nt, T)
        u_lagrange = u_analytical(lagrange_points)
        #transform lagrange basis to legendre basis
        trans = get_lagrange_to_legendre_matrix(ptrial, nt)
        u_legendre = trans @ u_lagrange
        D = pb.get_legendre_derivative_matrix(nt, ptrial, T)
        u_legendre_deriv = D @ u_legendre
        #get an evaluator
        from hmod.polynomial_bases import LegendreBasisEvaluator
        u_evaluator = LegendreBasisEvaluator(u_legendre, ptrial, nt, T)
        u_diff_evaluators = pb.LegendreBasisEvaluator(u_legendre_deriv, ptrial, nt, T)
        t_check = np.linspace(0+1e-4, T - 1e-4, 100) #to avoid evaluation directly at the end points
        u_vals_legendre = u_evaluator.evaluate(t_check)
        u_vals_expected = u_analytical(t_check)
        #assert that the difference is approximately zero
        diff_u = u_vals_expected - u_vals_legendre
        diff_u_norm = np.linalg.norm(diff_u)
        assert diff_u_norm < 1e-10
        du_vals_legendre = u_evaluator.evaluate_derivative(t_check)
        du_vals_expected = dt_u_analytical(t_check)
        diff_du = du_vals_expected - du_vals_legendre
        diff_du_norm = np.linalg.norm(diff_du)
        assert diff_du_norm < 1e-10
        du_vals_matrix = u_diff_evaluators.evaluate(t_check)
        diff_du_matr = du_vals_expected - du_vals_matrix
        diff_du_matr_norm = np.linalg.norm(diff_du_matr)
        assert diff_du_matr_norm < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])