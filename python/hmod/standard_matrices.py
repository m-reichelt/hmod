import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator
from hmod.polynomial_bases import get_lagrange_to_legendre_matrix


def legendre_extension_matrix_to_polynomial_degree(degree_original : int, degree_extended : int, nt : int):
    """Return the Legendre coefficient extension matrix.

    The matrix embeds a degree ``degree_original`` piecewise Legendre space
    into a degree ``degree_extended`` space on the same uniform mesh by
    appending zero coefficients for the higher polynomial modes.
    Coefficients use degree-major ordering:
    ``[degree 0 on all intervals, degree 1 on all intervals, ...]``.
    """
    ndof_original = (degree_original + 1)*nt
    ndof_extended = (degree_extended + 1)*nt
    if degree_extended < degree_original:
        exit("Error: cannot build extension matrix to lower or equal degree.")
    elif degree_extended == degree_original:
        return scipy.sparse.identity(ndof_original)
    else:
        #build the extension matrix as block matrix
        nrows = degree_extended+1
        ncols = degree_original+1
        I = scipy.sparse.identity(nt)
        Z = scipy.sparse.identity(nt)*0.0
        blocks = [[I if j==i else Z for j in range(ncols)] for i in range(nrows)]
        Ext = scipy.sparse.bmat(blocks).tocsr()
        return Ext

def get_legendre_legendre_matrix_for_kernel(polynomial_degree : int, nt : int, kernelMatrix : np.array):
    """Lift a local Legendre degree kernel to all time intervals.

    ``kernelMatrix`` contains the coupling between local Legendre degrees.
    The returned global matrix is ``kron(kernelMatrix, I_nt)`` and therefore
    acts on degree-major Legendre coefficient vectors.
    """
    I = scipy.sparse.identity(nt)
    from scipy.sparse import kron
    #just to be sure convert kernelMatrix to sparse before kronecker product
    kernelMatrix = scipy.sparse.csr_matrix(kernelMatrix)
    K = kron(kernelMatrix, I, format='csr')
    return K


def get_legendre_legendre_matrix_for_kernel_and_arbitrary_degrees(polynomial_degree_trial : int, polynomial_degree_test : int, nt : int, kernelMatrix : np.array):
    """Return a Legendre-Legendre matrix for possibly different degrees.

    Rows correspond to test functions and columns to trial functions. The
    supplied ``kernelMatrix`` is assembled on the maximum of trial and test
    degree and then restricted by Legendre extension matrices to the requested
    trial and test spaces.
    """
    polynomial_degree = max(polynomial_degree_trial, polynomial_degree_test)
    extension_trial = legendre_extension_matrix_to_polynomial_degree(polynomial_degree_trial, polynomial_degree, nt)
    extension_test = legendre_extension_matrix_to_polynomial_degree(polynomial_degree_test, polynomial_degree, nt)
    kernel = get_legendre_legendre_matrix_for_kernel(polynomial_degree, nt, kernelMatrix)
    K = extension_test.transpose() @ kernel @ extension_trial
    K = K.tocsr()
    return K

def get_kernel_legendre_legendre(polynomial_degree : int, derivatives_trial : int, derivatives_test : int,  nt : int, T : float):
    r"""Return the local Legendre kernel for derivative couplings.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the local
    kernel entries are scaled reference-cell integrals for

    ``<partial_t^i phi_m, partial_t^j psi_l>_I``.

    The row index ``l`` denotes the test Legendre degree and the column
    index ``m`` denotes the trial Legendre degree. Scaling by the interval
    length ``h = T / nt`` is included.
    """
    kernelMatrix = np.zeros((polynomial_degree+1, polynomial_degree+1))
    #compute without the transformation factors
    for l in range(polynomial_degree+1):
        for m in range(polynomial_degree+1):
            from scipy.special import legendre
            Pl = legendre(l)
            Pm = legendre(m)
            for _ in range(derivatives_test):
                Pl = np.polyder(Pl)
            for _ in range(derivatives_trial):
                Pm = np.polyder(Pm)
            integrand = np.polymul(Pl, Pm)
            integral = np.polyint(integrand)
            result = integral(1.0) - integral(-1.0)
            kernelMatrix[l,m] = result
    #compute factors for integrals and derivatives
    h = T/nt
    fac_derivative_trial = (2.0/h)**derivatives_trial
    fac_derivative_test = (2.0/h)**derivatives_test
    fac_integration = h/2.0
    total_factor = fac_derivative_trial * fac_derivative_test * fac_integration
    kernelMatrix = kernelMatrix * total_factor
    return kernelMatrix


def get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial : int, polynomial_degree_test : int
                                                 ,derivatives_trial : int, derivatives_test : int, nt : int, T : float):
    r"""Return a standard matrix in Legendre trial/test bases.

    Rows correspond to test basis functions and columns to trial basis
    functions. With ``i = derivatives_trial`` and
    ``j = derivatives_test``, the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, partial_t^j psi_r>_I``.

    Here ``phi_m`` is a Legendre trial basis function and ``psi_r`` is a
    Legendre test basis function.
    """
    polynomial_degree = max(polynomial_degree_trial, polynomial_degree_test)
    kernelMatrix = get_kernel_legendre_legendre(polynomial_degree, derivatives_trial, derivatives_test, nt, T)
    K = get_legendre_legendre_matrix_for_kernel_and_arbitrary_degrees(polynomial_degree_trial, polynomial_degree_test, nt, kernelMatrix)
    return K

def get_legendre_lagrange_matrix_for_derivatives(polynomial_degree_trial : int, polynomial_degree_test : int
                                                 ,derivatives_trial : int, derivatives_test : int, nt : int, T : float):
    r"""Return a standard matrix with Legendre trial and Lagrange test basis.

    Rows correspond to Lagrange test functions and columns to Legendre trial
    functions. With ``i = derivatives_trial`` and ``j = derivatives_test``,
    the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, partial_t^j psi_r>_I``.

    Here ``phi_m`` is a Legendre trial basis function and ``psi_r`` is a
    Lagrange test basis function.
    """
    #first get the transformation matrix from lagrange to legendre for the test functions
    Ttest = get_lagrange_to_legendre_matrix(polynomial_degree_test, nt)
    #get the legendre legendre matrix
    K_ll = get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial, polynomial_degree_test,
                                                        derivatives_trial, derivatives_test, nt, T)
    #put together the legendre lagrange matrix
    K = Ttest.transpose() @ K_ll
    K = K.tocsr()
    return K

def get_lagrange_legendre_matrix_for_derivatives(polynomial_degree_trial : int, polynomial_degree_test : int
                                                 ,derivatives_trial : int, derivatives_test : int, nt : int, T : float):
    r"""Return a standard matrix with Lagrange trial and Legendre test basis.

    Rows correspond to Legendre test functions and columns to Lagrange trial
    functions. With ``i = derivatives_trial`` and ``j = derivatives_test``,
    the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, partial_t^j psi_r>_I``.

    Here ``phi_m`` is a Lagrange trial basis function and ``psi_r`` is a
    Legendre test basis function.
    """
    #first get the transformation matrix from lagrange to legendre for the test functions
    Ttrial = get_lagrange_to_legendre_matrix(polynomial_degree_trial, nt)
    #get the legendre legendre matrix
    K_ll = get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial, polynomial_degree_test,
                                                        derivatives_trial, derivatives_test, nt, T)
    #put together the legendre lagrange matrix
    K = K_ll @ Ttrial
    K = K.tocsr()
    return K

def get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree_trial : int, polynomial_degree_test : int
                                                 ,derivatives_trial : int, derivatives_test : int, nt : int, T : float):
    r"""Return a standard matrix with Lagrange trial/test bases.

    Rows correspond to Lagrange test functions and columns to Lagrange trial
    functions. With ``i = derivatives_trial`` and ``j = derivatives_test``,
    the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, partial_t^j psi_r>_I``.

    This is the standard part of the temporal bilinear form. For example,
    ``derivatives_trial=1`` and ``derivatives_test=0`` gives the derivative
    term ``<partial_t u_h, v_h>_I``, while both derivative orders equal to
    zero give the mass term ``<u_h, v_h>_I``.
    """
    #first get the transformation matrix from lagrange to legendre for the trial function
    Ttrial = get_lagrange_to_legendre_matrix(polynomial_degree_trial, nt)
    #then get the respective legendre_lagrange matrix
    K_lagl = get_legendre_lagrange_matrix_for_derivatives(polynomial_degree_trial, polynomial_degree_test,
                                                          derivatives_trial, derivatives_test, nt, T)
    #put together the lagrange lagrange matrix
    K = K_lagl @ Ttrial
    K = K.tocsr()
    return K



def rhs_quadrature(f, nt : int, polynomial_degree_test : int, T : float, quad_order : int = None) -> np.ndarray:
    r"""Compute the Legendre load vector by quadrature.

    The returned vector contains entries

    ``b[r] = <f, psi_r>_I``,

    where ``psi_r`` are piecewise Legendre test basis functions of degree at
    most ``polynomial_degree_test``. The result uses degree-major ordering.
    """
    from numpy.polynomial.legendre import leggauss, legvander
    #get quadrature points and weights on [-1,1]
    if quad_order is not None:
        nqp = quad_order
    else:
        nqp = max(2, polynomial_degree_test+1)
    qp, qw = leggauss(nqp)
    h = T/nt
    #map quadrature points to all intervals at once
    t_qp = h * (np.arange(nt)[:, None] + 0.5*(qp[None, :] + 1.0))
    f_qp = np.asarray(f(t_qp.ravel()))
    if f_qp.shape == ():
        f_qp = np.full(t_qp.shape, f_qp)
    else:
        f_qp = np.squeeze(f_qp)
        if f_qp.shape != t_qp.shape:
            f_qp = np.broadcast_to(f_qp, (t_qp.size,)).reshape(t_qp.shape)

    legendre_values = legvander(qp, polynomial_degree_test)
    rhs_mat = ((f_qp * qw) @ legendre_values) * (h/2.0)

    rhs = rhs_mat.transpose().flatten()
    return rhs


def project_rhs_onto_legendre_basis(f, nt : int, polynomial_degree_test : int, T : float, quad_order : int = None) -> np.ndarray:
    r"""Return Legendre coefficients of the L2 projection of a right-hand side.

    This computes ``f_h`` in the piecewise Legendre space such that

    ``<f_h, psi_r>_I = <f, psi_r>_I``

    for all Legendre test basis functions ``psi_r`` of degree at most
    ``polynomial_degree_test``. The returned array contains the coefficients
    of ``f_h`` in degree-major ordering.
    """
    rhs_legendre = rhs_quadrature(f, nt, polynomial_degree_test, T, quad_order)
    M = get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree_test, polynomial_degree_test=polynomial_degree_test,
                                                     derivatives_trial=0, derivatives_test=0, nt=nt, T=T)
    rhs_legendre = scipy.sparse.linalg.spsolve(M, rhs_legendre)
    return rhs_legendre
