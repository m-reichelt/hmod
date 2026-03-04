import numpy as np
import pytest

import hmod.standard_matrices as sm
import hmod.hilbert_matrices as hm
import hmod.matrix_tools as mat_tools
import hmod.polynomial_bases as pb
import hmod.non_linear_operators as nops
from scipy.sparse.linalg import LinearOperator

def u_analytical(t):
    return np.sin(np.pi*t)

def dt_u_analytical(t):
    return np.pi * np.cos(np.pi*t)

def coeff(t):
    return 1.0+t*t

def f_analytical(t):
    return coeff(t)*dt_u_analytical(t)

T = 1.0
class ResidualEvaluator(LinearOperator):
    def __init__(self, nt, polynomial_degree_test):
        self.projection_degree = polynomial_degree_test+2 #this is exact for the chosen coefficient
        self.polynomial_degree_test = polynomial_degree_test
        self.nt = nt
        self.Op = nops.WeightedResidual(nops.ResidualType.Hilbert, polynomial_degree_test, self.projection_degree, nt, T)
        self.lagrange_to_legendre = sm.get_lagrange_to_legendre_matrix(polynomial_degree_test, nt)
        nrows = nt*polynomial_degree_test+1
        ncols = nrows
        shape = (nrows, ncols)
        super().__init__(dtype=None, shape=shape)


    def _matvec(self, x):
        x_legendre = self.lagrange_to_legendre @ x
        u_evaluator = pb.LegendreBasisEvaluator(x_legendre, self.polynomial_degree_test, self.nt, T)
        residual_evaluator = np.vectorize(lambda t: u_evaluator.evaluate_derivative(t)*coeff(t))
        result = self.Op.apply(residual_evaluator)
        return result


def solve_ode_only_hilbert_non_constant_coeff_directly(nt : int, polynomial_degree : int, number_of_modes : int, T : float):
    """Solve a simple ODE problem $c(t) u'(t) = f(t), u(0) = 0$ using only the Hilbert transformation as partial isometry.
        Mainly for testing purposes of the FFT implementation against L.

    Args:
        f_analytical (callable): The analytical right-hand side function.
        nt (int): Number of time intervals.
        polynomial_degree (int): Degree of the polynomial basis.
        number_of_modes (int): Number of modes in the Hilbert basis.
        T (float): Total time.

    Returns:
        np.ndarray: The solution vector.
    """
    #for rhs use the same polynomial degree
    polynomial_degree_rhs = polynomial_degree
    #project to piecewise polynomial space
    f_vec = sm.project_rhs_onto_legendre_basis(f_analytical, nt, polynomial_degree_rhs, T)
    #get the hilbert matrices
    A = ResidualEvaluator(nt, polynomial_degree)
    F = hm.Operator_I_H_Legendre_Lagrange(number_of_modes, nt, polynomial_degree_rhs, polynomial_degree)
    rhs_vec = np.array(F @ f_vec)
    #transfer the system to a dense matrix for direct solving
    Kd = mat_tools.linear_operator_to_matrix(A)
    #homogenize
    K0 = Kd[1:,1:]
    rhs0 = rhs_vec[1:]
    #solve the system
    u0 = np.linalg.solve(K0, rhs0)
    # add the initial condition back
    u_vec = np.zeros(Kd.shape[0])
    u_vec[1:] = u0
    #compute errors
    legendre_trans = pb.get_lagrange_to_legendre_matrix(polynomial_degree, nt)
    u_vec_legendre = legendre_trans @ u_vec
    legendre_sol = pb.LegendreBasisEvaluator(u_vec_legendre, polynomial_degree, nt, T)

    from hmod.norms import compute_l2_norm, compute_h12_seminorm

    error_integrand = lambda t: (legendre_sol.evaluate(t) - u_analytical(t))

    L2_error = compute_l2_norm(error_integrand, nt, T)
    H1_2_error = compute_h12_seminorm(error_integrand, T, quad_tol = 1e-8)

    return L2_error, H1_2_error


def test_hilbert_only_ode_convergence():
    T = 1.0
    polynomial_degrees = [1,2,3,4]
    nt_values = [512, 256, 128, 64, 32]
    nt_values = [int(n/8) for n in nt_values]  #we will double nt in the convergence test
    number_of_modes = int(1e5)
    for (p,n) in zip(polynomial_degrees, nt_values):
        l2e0, h12e_0 = solve_ode_only_hilbert_non_constant_coeff_directly(n, p, number_of_modes, T)
        l2e1, h12e_1 = solve_ode_only_hilbert_non_constant_coeff_directly(n*2, p, number_of_modes, T)
        eocl2 = np.log2(l2e0/l2e1)
        eoch12 = np.log2(h12e_0/h12e_1)
        #check that eocl2 is approximately p+1
        expected_eocl2 = p + 1
        expected_eoch12 = p + 0.5
        tol = 0.1
        assert abs(eocl2 - expected_eocl2) < tol, f"L2 EOC {eocl2} deviates from expected {expected_eocl2} for p={p}"
        assert abs(eoch12 - expected_eoch12) < tol, f"H1/2 EOC {eoch12} deviates from expected {expected_eoch12} for p={p}"


if __name__ == "__main__":
    test_hilbert_only_ode_convergence()
    #pytest.main([__file__])