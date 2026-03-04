import pytest
from ngsolve import *
from ngsolve.meshes import Make1DMesh


def ngsolve_mass_matrix(tpoints):

    nt = len(tpoints) - 1
    T = tpoints[-1]
    mapping = lambda t : t*T
    mesh = Make1DMesh(n=nt, mapping=mapping)
    V = H1(mesh, order=1)  # Linear elements with zero at t=0

    a = BilinearForm(V)
    a += InnerProduct(V.TrialFunction(), V.TestFunction()) * dx
    a.Assemble()

    rows,cols,vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.csr_matrix((vals,(rows,cols)))

    return M

def ngsolve_mass_p0_p1_matrix(tpoints):

    nt = len(tpoints) - 1
    T = tpoints[-1]
    mapping = lambda t : t*T
    mesh = Make1DMesh(n=nt, mapping=mapping)
    V = H1(mesh, order=1)  # Linear elements with zero at t=0
    W = L2(mesh, order=0)  # Constant elements

    a = BilinearForm(W, V)
    a += InnerProduct(W.TrialFunction(), V.TestFunction()) * dx
    a.Assemble()

    rows,cols,vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.csr_matrix((vals,(rows,cols)))

    return M

def ngsolve_mass_p0_matrix(tpoints):

    nt = len(tpoints) - 1
    T = tpoints[-1]
    mapping = lambda t : t*T
    mesh = Make1DMesh(n=nt, mapping=mapping)
    W = L2(mesh, order=0)  # Constant elements

    a = BilinearForm(W)
    a += InnerProduct(W.TrialFunction(), W.TestFunction()) * dx
    a.Assemble()

    rows,cols,vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.csr_matrix((vals,(rows,cols)))

    return M

def ngsolve_mass_p0_quadrature(f, tpoints, qorder):

    nt = len(tpoints) - 1
    T = tpoints[-1]
    mapping = lambda t : t*T
    mesh = Make1DMesh(n=nt, mapping=mapping)
    W = L2(mesh, order=0)  # Constant elements

    lf = LinearForm(W)
    lf += f(x)*W.TestFunction()*dx(bonus_intorder=qorder)
    lf.Assemble()

    return lf.vec.FV().NumPy()



def ngsolve_dt_I(tpoints):

    nt = len(tpoints) - 1
    T = tpoints[-1]
    mapping = lambda t : t*T
    mesh = Make1DMesh(n=nt, mapping=mapping)
    V = H1(mesh, order=1)  # Linear elements with zero at t=0

    a = BilinearForm(V)
    u, v = V.TnT()
    a += InnerProduct(grad(u), v) * dx
    a.Assemble()

    rows,cols,vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.csr_matrix((vals,(rows,cols)))

    return M

def test_quadrature_p0():
    from hmod.standard_matrices import rhs_quadrature
    import numpy as np
    nt = 10
    T = 3.0
    polynomial_degree = 0
    qorder = 10
    tpoints = np.linspace(0, T, nt+1)
    f_ngs = lambda t : t**3+ 5.0
    f_legendre = lambda t : t**3+ 5.0
    rhs_ngs = ngsolve_mass_p0_quadrature(f_ngs, tpoints, qorder)
    rhs_leg = rhs_quadrature(f_legendre, nt, 0, T, quad_order=qorder)
    diff_vec = rhs_ngs - rhs_leg
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-12

def test_rhs_p1():
    from hmod.standard_matrices import get_legendre_lagrange_matrix_for_derivatives
    import numpy as np
    nt = 10
    T = 3.0
    polynomial_degree = 1
    tpoints = np.linspace(0, T, nt+1)
    M_ngsolve = np.array(ngsolve_mass_p0_p1_matrix(tpoints).todense())
    M_via_legendre = get_legendre_lagrange_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree-1, polynomial_degree_test=polynomial_degree,
                                                                  derivatives_trial=0, derivatives_test=0, nt=nt, T=T)
    M_via_legendre_dense = np.array(M_via_legendre.todense())
    diff = np.linalg.norm(M_ngsolve - M_via_legendre_dense)
    assert diff < 1e-12

def test_p1_mass():
    from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives
    import numpy as np
    nt = 10
    T = 3.0
    polynomial_degree = 1
    tpoints = np.linspace(0, T, nt+1)
    M_ngsolve = np.array(ngsolve_mass_matrix(tpoints).todense())
    M_via_legendre = get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree, polynomial_degree_test=polynomial_degree
                                                                  ,derivatives_trial=0, derivatives_test=0, nt=nt, T=T)
    M_via_legendre_dense = np.array(M_via_legendre.todense())
    diff = np.linalg.norm(M_ngsolve - M_via_legendre_dense)
    assert diff < 1e-12

def test_p0_mass():
    from hmod.standard_matrices import get_legendre_legendre_matrix_for_derivatives
    import numpy as np
    nt = 10
    T = 3.0
    polynomial_degree = 0
    tpoints = np.linspace(0, T, nt+1)
    M_ngsolve = np.array(ngsolve_mass_p0_matrix(tpoints).todense())
    M_via_legendre = get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree, polynomial_degree_test=polynomial_degree
                                                                  ,derivatives_trial=0, derivatives_test=0, nt=nt, T=T)
    M_via_legendre_dense = np.array(M_via_legendre.todense())
    diff = np.linalg.norm(M_ngsolve - M_via_legendre_dense)
    assert diff < 1e-12

def test_p1_dt_I():
    from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives
    import numpy as np
    nt = 3
    T = 2.0
    polynomial_degree = 1
    tpoints = np.linspace(0, T, nt+1)
    M_ngsolve = np.array(ngsolve_dt_I(tpoints).todense())
    M_via_legendre = get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree, polynomial_degree_test=polynomial_degree
                                                                  ,derivatives_trial=1, derivatives_test=0, nt=nt, T=T)
    M_via_legendre_dense = np.array(M_via_legendre.todense())
    diff = np.linalg.norm(M_ngsolve - M_via_legendre_dense)
    assert diff < 1e-12


def test_derivative_matrix():
    from hmod.standard_matrices import get_lagrange_legendre_matrix_for_derivatives
    from hmod.polynomial_bases import get_langrange_points
    import numpy as np
    nt = 19
    T = 3.0
    max_degree= 4
    pol_degrees_test = [k for k in range(0,max_degree)]
    pol_degrees_trial = [k+1 for k in range(0,max_degree)]
    for (ptrial, ptest) in zip(pol_degrees_trial, pol_degrees_test):
        #define a polynomial of of degree ptrial, so the derivative must be exact
        coeffs = [1.0 for _ in range(ptrial+1)]
        u_analytical = np.poly1d(coeffs)
        dt_u_analytical = u_analytical.deriv()
        #define a polynomial function
        lagrange_points = get_langrange_points(ptrial, nt, T)
        u_lagrange = u_analytical(lagrange_points)
        #get the derivative matrix
        D = get_lagrange_legendre_matrix_for_derivatives(ptrial, ptest, 1, 0, nt, T)
        Dd = np.array(D.todense())
        du_vec_rhs = D @ u_lagrange
        #get mass for projection
        from hmod.standard_matrices import get_legendre_legendre_matrix_for_derivatives
        M_legendre = get_legendre_legendre_matrix_for_derivatives(ptest, ptest, 0, 0, nt, T)
        M_legendre_dense = np.array(M_legendre.todense())
        #solve for du coefficients in legendre basis
        from numpy.linalg import solve
        du_vec = solve(M_legendre_dense, du_vec_rhs)
        #get an evaluator
        from hmod.polynomial_bases import LegendreBasisEvaluator
        du_evaluator = LegendreBasisEvaluator(du_vec, ptest, nt, T)
        t_check = np.linspace(0, T, 100)
        for t in t_check:
            #check that u_langrange evaluated at t is the same as u_analytical(t)
            from hmod.polynomial_bases import get_lagrange_to_legendre_matrix
            trans = get_lagrange_to_legendre_matrix(ptrial, nt)
            u_leg = trans @ u_lagrange
            legendre_evaluator = LegendreBasisEvaluator(u_leg, ptrial, nt, T)
            u_eval = legendre_evaluator.evaluate(t)
            u_exact = u_analytical(t)
            diff = abs(u_eval - u_exact)
            assert diff < 1e-10
        du_eval = du_evaluator.evaluate(t_check)
        du_exact = dt_u_analytical(t_check)
        diff = np.linalg.norm(du_eval - du_exact)
        assert diff < 1e-10




if __name__ == "__main__":
    pytest.main([__file__])