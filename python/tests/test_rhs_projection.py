import pytest
def tau_trans(t):
    return 2.0*t - 1.0

def legendre_polynomial(n, x):
    from scipy.special import legendre
    Pn = legendre(n)
    return Pn(x)

def test_rhs_projection_of_legendre_pols():
    from hmod.standard_matrices import project_rhs_onto_legendre_basis
    n_max = 10
    for n in range(n_max+1):
        f_legendre = lambda t : legendre_polynomial(n, tau_trans(t))
        #project onto legendre basis on just one element
        nt = 1
        polynomial_degree_rhs = n #use same degree for rhs
        T = 1.0
        projected_vals = project_rhs_onto_legendre_basis(f_legendre, nt, polynomial_degree_rhs, T=T)
        #check that the projected values are zero except for the nth entry
        for k in range(polynomial_degree_rhs+1):
            if k == n:
                assert abs(projected_vals[k] - 1.0) < 1e-10
            else:
                assert abs(projected_vals[k]) < 1e-10


def general_polynomial(t, degree):
    import numpy as np
    coeffs = [1.0]*(degree+1)
    p = np.poly1d(coeffs)
    return p(t)

def evaluate_lagrange_basis(t, degree, nt, dofs, T):
    import numpy as np
    from scipy.special import legendre
    #reshape dofs
    legendre_dofs = dofs.reshape((degree+1, nt)).transpose()
    res = 0.
    #find element for T
    h = T/nt
    res = 0.0
    for ie in range(nt):
        t_left = ie*h
        t_right = (ie+1)*h
        if t >= t_left and t <= t_right:
            #map t to reference element
            xi = 2.0*(t - t_left)/h - 1.0
            #evaluate lagrange basis on reference element
            for j in range(degree+1):
                res += legendre(j)(xi) * legendre_dofs[ie, j]
            break
    return res

def test_rhs_projection_of_polynomial():
    from hmod.standard_matrices import project_rhs_onto_legendre_basis, rhs_quadrature
    from hmod.polynomial_bases import LegendreBasis
    import numpy as np
    n_max = 3
    for n in range(n_max+1):
        f_legendre = lambda t : general_polynomial(t, n)
        #project onto legendre basis on just one element
        nt = 2
        h = 1.0/nt
        polynomial_degree_rhs = n #use same degree for rhs
        T = 1.0
        projected_vals = project_rhs_onto_legendre_basis(f_legendre, nt, polynomial_degree_rhs, T=T)
        #get a basis an check, that values match for same degree
        legendre_basis = LegendreBasis(polynomial_degree_rhs, nt, T=T)
        #reconstruct the polynomial from the projected values
        reconstructed_poly = lambda t : legendre_basis.evaluate(t, projected_vals)
        reconstructed_poly_manual = lambda t : evaluate_lagrange_basis(t, polynomial_degree_rhs, nt, projected_vals, T)
        #get quadrature of rhs
        rhs_quad = rhs_quadrature(f_legendre, nt, polynomial_degree_rhs, T=T, quad_order=20)/h #normalize by h
        rhs_quad_eval = lambda t : evaluate_lagrange_basis(t, polynomial_degree_rhs, nt, rhs_quad, T)
        #check at several points
        test_points = np.linspace(0, T, 100)
        plotting = False
        if plotting:
            import matplotlib.pyplot as plt
            f_vals = [f_legendre(tp) for tp in test_points]
            rec_vals = [reconstructed_poly(tp) for tp in test_points]
            rec_vals_manual = [reconstructed_poly_manual(tp) for tp in test_points]
            plt.plot(test_points, f_vals, label="Original")
            plt.plot(test_points, rec_vals, '--', label="Reconstructed")
            plt.plot(test_points, rec_vals_manual, ':', label="Reconstructed manual")
            plt.plot(test_points, [rhs_quad_eval(tp) for tp in test_points], '-.', label="RHS quadrature")
            plt.legend()
            plt.title(f"Degree {n} polynomial projection")
            plt.show()
        for tp in test_points:
            original_val = f_legendre(tp)
            reconstructed_val = reconstructed_poly(tp)
            assert abs(original_val - reconstructed_val) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])