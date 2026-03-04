import numpy as np
import pytest
import scipy

import hmod.standard_matrices as sm
import hmod.hilbert_matrices as hm
import hmod.matrix_tools as mat_tools
import hmod.polynomial_bases as pb

def u_analytical(t):
    return np.sin(np.pi*t)

def solve_projection_legendre(nt : int, polynomial_degree : int, number_of_modes : int, T : float):
    """Solve a simple ODE problem $u'(t) = f(t), u(0) = 0$ using only the Hilbert transformation as partial isometry.
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
    u_vec = sm.project_rhs_onto_legendre_basis(u_analytical, nt, polynomial_degree_rhs, T)
    legendre_sol = pb.LegendreBasisEvaluator(u_vec, polynomial_degree, nt, T)

    from hmod.norms import compute_l2_norm, compute_h12_seminorm

    error_integrand = lambda t: (legendre_sol.evaluate(t) - u_analytical(t))

    L2_error = compute_l2_norm(error_integrand, nt, T)

    return L2_error

def solve_projection_lagrange(nt : int, polynomial_degree : int, number_of_modes : int, T : float):
    """Solve a simple ODE problem $u'(t) = f(t), u(0) = 0$ using only the Hilbert transformation as partial isometry.
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
    u_vec_legendre = sm.project_rhs_onto_legendre_basis(u_analytical, nt, polynomial_degree_rhs, T)
    M_leg_lag = sm.get_legendre_lagrange_matrix_for_derivatives(polynomial_degree, polynomial_degree, 0, 0, nt, T)
    u_vec_rhs  = M_leg_lag @ u_vec_legendre
    #solve for lagrange coefficients
    M_lag_lag = sm.get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree, polynomial_degree, 0, 0, nt, T)
    u_vec = scipy.sparse.linalg.spsolve(M_lag_lag, u_vec_rhs)

    #get the solution back in legendre basis
    trans = pb.get_lagrange_to_legendre_matrix(polynomial_degree, nt)
    u_sol_leg = trans @ u_vec

    legendre_sol = pb.LegendreBasisEvaluator(u_sol_leg, polynomial_degree, nt, T)

    from hmod.norms import compute_l2_norm, compute_h12_seminorm

    error_integrand = lambda t: (legendre_sol.evaluate(t) - u_analytical(t))

    L2_error = compute_l2_norm(error_integrand, nt, T)

    return L2_error


def test_standard_mass_convergence_legendre():
    T = 1.0
    polynomial_degrees = [1,2,3,4]
    nt_values = [512, 256, 128, 64, 32]
    number_of_modes = int(1e5)
    for (p,n) in zip(polynomial_degrees, nt_values):
        l2e0 = solve_projection_legendre(n, p, number_of_modes, T)
        l2e1 = solve_projection_legendre(n*2, p, number_of_modes, T)
        eocl2 = np.log2(l2e0/l2e1)
        #check that eocl2 is approximately p+1
        expected_eocl2 = p + 1
        tol = 0.1
        assert abs(eocl2 - expected_eocl2) < tol, f"L2 EOC {eocl2} deviates from expected {expected_eocl2} for p={p}"

def test_standard_mass_convergence_lagrange():
    T = 1.0
    polynomial_degrees = [1,2,3,4]
    nt_values = [512, 256, 128, 64, 32]
    number_of_modes = int(1e5)
    for (p,n) in zip(polynomial_degrees, nt_values):
        l2e0 = solve_projection_lagrange(n, p, number_of_modes, T)
        l2e1 = solve_projection_lagrange(n*2, p, number_of_modes, T)
        eocl2 = np.log2(l2e0/l2e1)
        #check that eocl2 is approximately p+1
        expected_eocl2 = p + 1
        tol = 0.1
        assert abs(eocl2 - expected_eocl2) < tol, f"L2 EOC {eocl2} deviates from expected {expected_eocl2} for p={p}"


if __name__ == "__main__":
    pytest.main([__file__])