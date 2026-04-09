import numpy as np
import scipy.sparse as sp
class LegendreBasis:
    def __init__(self, polynomial_degree : int, nt : int, T : float):
        from hmod.hmod import LegendreBasis
        self.polynomial_degree = polynomial_degree
        self.nt = nt
        self.T = T
        self._legendre_basis = LegendreBasis(polynomial_degree, nt, T)

    def evaluate(self, t, legendre_vals : np.ndarray):
        """Evaluate the Legendre polynomial of given degree at x."""
        evaluator = np.vectorize(lambda t : self._legendre_basis.evaluate_at(t, legendre_vals))
        return evaluator(t)

    def evaluate_all_basis_functions(self, t):
        """Evaluate all Legendre basis functions of given degree at x."""
        basis_values = np.array(self._legendre_basis.evaluate_all_basis_functions(t))
        return basis_values

    def evaluate_all_basis_function_derivatives(self, t):
        """Evaluate all Legendre basis functions of given degree at x."""
        basis_values = np.array(self._legendre_basis.evaluate_all_basis_function_derivatives(t))
        return basis_values

class LegendreBasisEvaluator:
    """Class to evaluate Legendre basis functions at given points.
        Recommended for multiple evaluations with the same DOFs to avoid redundant setup costs.
    """
    def __init__(self, dofs : np.ndarray ,polynomial_degree : int, nt : int, T : float):
        from hmod.hmod import LegendreBasisEvaluator
        self.polynomial_degree = polynomial_degree
        self.nt = nt
        self.T = T
        self._legendre_basis_evaluator = LegendreBasisEvaluator(polynomial_degree, nt, T)
        self._legendre_basis_evaluator.set_dofs(dofs)

    def set_dofs(self, dofs : np.ndarray):
        """Set the DOFs for the evaluator."""
        self._legendre_basis_evaluator.set_dofs(dofs)

    def evaluate(self, t):
        """Evaluate the Legendre polynomial of given degree at x."""
        evaluator = np.vectorize(lambda t : self._legendre_basis_evaluator.evaluate_at(t))
        return evaluator(t)

    def evaluate_derivative(self, t):
        """Evaluate the Legendre polynomial of given degree at x."""
        evaluator = np.vectorize(lambda t : self._legendre_basis_evaluator.evaluate_derivative_at(t))
        return evaluator(t)



def get_lagrange_to_legendre_matrix(polynomial_degree : int, nt : int):
    from hmod.hmod import lagrange_to_legendre_basis_transformation
    from hmod.matrix_tools import triplets_to_linear_operator
    ri, ci, vals = lagrange_to_legendre_basis_transformation(polynomial_degree, nt)
    return triplets_to_linear_operator(ri, ci, vals)

def get_langrange_points(polynomial_degree : int, nt : int, T : float):
    from hmod.hmod import get_lagrange_points
    points = get_lagrange_points(polynomial_degree, nt, T)
    return np.array(points)



def _get_refinement_matrices_on_interval(polynomial_degree : int):
    from scipy.special import legendre
    left_child_mat = np.zeros((polynomial_degree+1, polynomial_degree+1))
    right_child_mat = np.zeros((polynomial_degree+1, polynomial_degree+1))
    trans_left = np.poly1d(np.array([0.5, -0.5]))
    trans_right = np.poly1d(np.array([0.5, 0.5]))
    for k in range(polynomial_degree+1):
        Pk = legendre(k)
        for m in range(polynomial_degree+1):
            Pm = legendre(m)
            P_left = np.polyval(Pm, trans_left)
            P_right = np.polyval(Pm, trans_right)
            integrand_left = np.polymul(Pk, P_left)
            integrand_right = np.polymul(Pk, P_right)
            #compute coefficients via integral
            integral_left = np.polyint(integrand_left)(1) - np.polyint(integrand_left)(-1)
            integral_right = np.polyint(integrand_right)(1) - np.polyint(integrand_right)(-1)
            coeff_left = (2*k + 1)/2 * integral_left
            coeff_right = (2*k + 1)/2 * integral_right
            left_child_mat[k,m] = coeff_left
            right_child_mat[k,m] = coeff_right

    return left_child_mat, right_child_mat


def legendre_refinement_matrix(polynomial_degree : int , nt_coarse : int):
    """Get the refinement matrix from coarse to once refined grid in Legendre basis."""
    from scipy.sparse import identity, kron, csr_matrix, bmat
    left_child_mat, right_child_mat = _get_refinement_matrices_on_interval(polynomial_degree)
    def T_ij(i : int, j : int):
        #get the transformation matrix with target u^i from u^j (basis vectors in Legendre basis of respective degree)
        val_left = left_child_mat[i,j]
        val_right = right_child_mat[i,j]
        I = identity(nt_coarse)
        local_trans = np.array([[val_left],[val_right]])
        local_trans = csr_matrix(local_trans)
        return kron(I, local_trans)

    blocks = [[T_ij(i,j) for j in range(polynomial_degree+1)] for i in range(polynomial_degree+1)]
    refinement_matrix = bmat(blocks)
    refinement_matrix = refinement_matrix.tocsr()
    return refinement_matrix

def get_lagrange_prolongation_matrix(nt_coarse : int, nt_fine : int, polynomial_degree : int):
    """Get the prolongation matrix from coarse to fine grid in Lagrange basis."""
    #todo: this routine has at the moment O(N^2) complexity and could be improved, but like this it is simple and clear
    T = 1.0 #irrelevant for the matrix itself
    lagrange_points_coarse = get_langrange_points(polynomial_degree, nt_coarse, T)
    lagrange_points_fine = get_langrange_points(polynomial_degree, nt_fine, T)
    trans = get_lagrange_to_legendre_matrix(polynomial_degree, nt_coarse)
    P = np.zeros((len(lagrange_points_fine), len(lagrange_points_coarse)))
    for i in range(len(lagrange_points_coarse)):
        e_vec = np.zeros(len(lagrange_points_coarse))
        e_vec[i] = 1.0
        legendre_dofs = trans @ e_vec
        legendre_evaluator = LegendreBasisEvaluator(legendre_dofs, polynomial_degree, nt_coarse, T)
        col = legendre_evaluator.evaluate(lagrange_points_fine)
        P[:,i] = col

    #transform to csr matrix
    from scipy.sparse import csr_matrix
    P_csr = csr_matrix(P)
    #make sure that the zeros are cut out
    P_csr.eliminate_zeros()
    return P_csr


def legendre_derivative_matrix(num_intervals: int, p: int, h: float, square: bool = False):
    """
    Global sparse matrix for d/dt on piecewise-Legendre coefficients
    with degree-major ordering:
        [u_0(all intervals), u_1(all intervals), ..., u_p(all intervals)]^T

    Uses the standard Legendre basis P_n on each interval.

    Parameters
    ----------
    num_intervals : int
        Number of equal intervals.
    p : int
        Polynomial degree on each interval.
    h : float
        Interval size.
    square : bool
        If False, return the natural rectangular map into degree <= p-1.
        If True, append one zero block-row to make the matrix square.
    """
    nrows_deg = p + 1 if square else p
    ncols_deg = p + 1

    blocks = [[None for _ in range(ncols_deg)] for _ in range(nrows_deg)]

    scale = 2.0 / h
    I = sp.eye(num_intervals, format="csr")

    for n in range(ncols_deg):          # input degree
        for m in range(min(n, nrows_deg)):   # output degree
            if (n - m) % 2 == 1:
                blocks[m][n] = scale * (2*m + 1) * I

    return sp.bmat(blocks, format="csr")


if __name__ == "__main__":
    mat = _get_refinement_matrices_on_interval(2)
    tmp = 0















