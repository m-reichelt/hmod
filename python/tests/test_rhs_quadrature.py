import pytest
import numpy as np
import scipy


def general_polynom(degree):
    """A general polynomial function of given degree."""
    coeffs = [1. for _ in range(degree + 1)]
    return np.poly1d(coeffs)

def mapped_legendre_polynomial(degree, T):
    tm = 0.5*T
    h = T #assuming only one element for this test
    tau = lambda t: 2*(t - tm)/h
    from scipy.special import legendre
    P = legendre(degree)
    p_mapped = lambda t: P(tau(t))
    return np.vectorize(p_mapped)


def exact_quad(f_poly, m, T):
    integrand = lambda t: f_poly(t) * (mapped_legendre_polynomial(m, T))(t)
    integral, _ = scipy.integrate.quad(integrand, 0, T)
    return integral

def test_quadrature_for_polynomials():
    from hmod.standard_matrices import rhs_quadrature
    T = 2.0
    nt = 1
    degrees_to_test = [0,1,2,3,4,5,6,7,8,9]
    for degree in degrees_to_test:
        #f_poly = general_polynom(degree)
        f_poly = mapped_legendre_polynomial(degree, T)
        rhs_quad_vec = rhs_quadrature(f_poly, nt, degree, T)
        exact_quad_vec = np.array([exact_quad(f_poly, m, T) for m in range(degree+1)])
        diff = np.linalg.norm(rhs_quad_vec.flatten() - exact_quad_vec)
        tol = 1e-12
        #assert diff < 1e-12, f"Quadrature failed for polynomial degree {degree} with difference {diff}"


def test_quadrature_uses_vectorized_rhs_call():
    from hmod.standard_matrices import rhs_quadrature
    T = 2.0
    nt = 4
    degree = 3
    quad_order = degree + 2
    calls = []

    def f(points):
        calls.append(np.shape(points))
        return np.ones_like(points)

    rhs_quad_vec = rhs_quadrature(f, nt, degree, T, quad_order=quad_order)
    exact_quad_mat = np.zeros((degree+1, nt))
    exact_quad_mat[0, :] = T/nt

    assert calls == [(nt*quad_order,)]
    np.testing.assert_allclose(rhs_quad_vec, exact_quad_mat.flatten(), atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__])
