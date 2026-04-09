import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator
from hmod.polynomial_bases import get_lagrange_to_legendre_matrix


def legendre_extension_matrix_to_polynomial_degree(degree_original : int, degree_extended : int, nt : int):
    """Get the extension matrix from original degree to extended degree for the Legendre Basis."""
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
    """Get the matrix for a given kernel in the legendre basis.
        The kernel is for a(p_lm, p_ln), so it can still incorporate a factor for the transformation
    """
    I = scipy.sparse.identity(nt)
    from scipy.sparse import kron
    #just to be sure convert kernelMatrix to sparse before kronecker product
    kernelMatrix = scipy.sparse.csr_matrix(kernelMatrix)
    K = kron(kernelMatrix, I, format='csr')
    return K


def get_legendre_legendre_matrix_for_kernel_and_arbitrary_degrees(polynomial_degree_trial : int, polynomial_degree_test : int, nt : int, kernelMatrix : np.array):
    """Get the matrix for a given kernel in the legendre basis.
        The kernel is for a(p_lm, p_ln), so it can still incorporate a factor for the transformation
        This function allows for different polynomial degrees in test and trial.
    """
    polynomial_degree = max(polynomial_degree_trial, polynomial_degree_test)
    extension_trial = legendre_extension_matrix_to_polynomial_degree(polynomial_degree_trial, polynomial_degree, nt)
    extension_test = legendre_extension_matrix_to_polynomial_degree(polynomial_degree_test, polynomial_degree, nt)
    kernel = get_legendre_legendre_matrix_for_kernel(polynomial_degree, nt, kernelMatrix)
    K = extension_test.transpose() @ kernel @ extension_trial
    K = K.tocsr()
    return K

def get_kernel_legendre_legendre(polynomial_degree : int, derivatives_trial : int, derivatives_test : int,  nt : int, T : float):
    """Get the kernel matrix for the legendre basis."""
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
    """Get the legendre legendre matrix for given derivatives and polynomial degrees."""
    polynomial_degree = max(polynomial_degree_trial, polynomial_degree_test)
    kernelMatrix = get_kernel_legendre_legendre(polynomial_degree, derivatives_trial, derivatives_test, nt, T)
    K = get_legendre_legendre_matrix_for_kernel_and_arbitrary_degrees(polynomial_degree_trial, polynomial_degree_test, nt, kernelMatrix)
    return K

def get_legendre_lagrange_matrix_for_derivatives(polynomial_degree_trial : int, polynomial_degree_test : int
                                                 ,derivatives_trial : int, derivatives_test : int, nt : int, T : float):
    """Get the legendre lagrange matrix for given derivatives and polynomial degrees.
       In this sense the trial functions are in legendre basis and the test functions in lagrange basis.
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
    """Get the lagrange legendre matrix for given derivatives and polynomial degrees.
       In this sense the trial functions are in lagrange basis and the test functions in legendre basis.
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
    """Get the lagrange lagrange matrix for given derivatives and polynomial degrees.
       In this sense the trial functions are in lagrange basis and the test functions in lagrange basis.
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
    """Compute the rhs vector by quadrature."""
    from numpy.polynomial.legendre import leggauss
    from scipy.special import legendre
    #get quadrature points and weights on [-1,1]
    if quad_order is not None:
        nqp = quad_order
    else:
        nqp = max(2, polynomial_degree_test+1)
    qp, qw = leggauss(nqp)
    h = T/nt
    rhs_mat = np.zeros((nt, polynomial_degree_test+1))
    for ie in range(nt):
        #map quadrature points to [t_ie, t_ie+1]
        t_left = ie*h
        t_right = (ie+1)*h
        t_qp = 0.5*( (t_right - t_left)*qp + (t_right + t_left) )
        f_qp = f(t_qp)
        for j in range(polynomial_degree_test+1):
            #evaluate the j-th legendre polynomial at the quadrature points
            Pj = legendre(j)
            Pj_qp = Pj( qp )  #evaluate at quadrature points on [-1,1]
            integral = np.sum( qw * f_qp * Pj_qp ) * (h/2.0)
            rhs_mat[ie, j] = integral

    rhs = rhs_mat.transpose().flatten()
    return rhs


def project_rhs_onto_legendre_basis(f, nt : int, polynomial_degree_test : int, T : float, quad_order : int = None) -> np.ndarray:
    """Project the rhs function onto the legendre basis using quadrature."""
    rhs_legendre = rhs_quadrature(f, nt, polynomial_degree_test, T, quad_order)
    M = get_legendre_legendre_matrix_for_derivatives(polynomial_degree_trial=polynomial_degree_test, polynomial_degree_test=polynomial_degree_test,
                                                     derivatives_trial=0, derivatives_test=0, nt=nt, T=T)
    rhs_legendre = scipy.sparse.linalg.spsolve(M, rhs_legendre)
    return rhs_legendre