import numpy as np
import scipy.sparse as sp


def _as_1d_coefficients(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values


class LegendreBasis:
    def __init__(self, polynomial_degree : int, nt : int, T : float):
        from hmod.hmod import LegendreBasis
        self.polynomial_degree = polynomial_degree
        self.nt = nt
        self.T = T
        self._legendre_basis = LegendreBasis(polynomial_degree, nt, T)

    def evaluate(self, t, legendre_vals : np.ndarray):
        """Evaluate the Legendre polynomial of given degree at x."""
        legendre_vals = _as_1d_coefficients(legendre_vals)
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
        self.set_dofs(dofs)

    def set_dofs(self, dofs : np.ndarray):
        """Set the DOFs for the evaluator."""
        dofs = _as_1d_coefficients(dofs)
        self._legendre_basis_evaluator.set_dofs(dofs)

    def evaluate(self, t):
        """Evaluate the Legendre polynomial of given degree at x."""
        t = np.asarray(t, dtype=float)
        values = self._legendre_basis_evaluator.evaluate_many(np.ascontiguousarray(t.ravel()))
        return np.asarray(values).reshape(t.shape)

    def evaluate_derivative(self, t):
        """Evaluate the Legendre polynomial of given degree at x."""
        t = np.asarray(t, dtype=float)
        values = self._legendre_basis_evaluator.evaluate_derivative_many(np.ascontiguousarray(t.ravel()))
        return np.asarray(values).reshape(t.shape)



def get_lagrange_to_legendre_matrix(
    polynomial_degree: int,
    nt: int,
    periodic: bool = False,
):
    """Map continuous Lagrange DOFs to degree-major Legendre coefficients.

    A non-periodic Lagrange vector has ``nt*polynomial_degree + 1`` entries.
    With ``periodic=True``, the endpoint at ``T`` is identified with the
    endpoint at ``0`` and the vector has ``nt*polynomial_degree`` entries.
    """
    if periodic and polynomial_degree < 1:
        raise ValueError("periodic Lagrange spaces require polynomial_degree >= 1")
    if periodic and nt < 1:
        raise ValueError("periodic Lagrange spaces require nt >= 1")

    from hmod.hmod import lagrange_to_legendre_basis_transformation
    from scipy.sparse import coo_matrix

    ri, ci, vals = lagrange_to_legendre_basis_transformation(
        polynomial_degree, nt, periodic
    )
    n_rows = (polynomial_degree + 1) * nt
    n_cols = polynomial_degree * nt + (0 if periodic else 1)
    return coo_matrix(
        (vals, (ri, ci)), shape=(n_rows, n_cols)
    ).tocsr()


def get_lagrange_points(
    polynomial_degree: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    """Return global Lagrange points, omitting ``T`` in the periodic space."""
    if periodic and polynomial_degree < 1:
        raise ValueError("periodic Lagrange spaces require polynomial_degree >= 1")
    if periodic and nt < 1:
        raise ValueError("periodic Lagrange spaces require nt >= 1")

    from hmod.hmod import get_lagrange_points as rust_get_lagrange_points

    points = rust_get_lagrange_points(polynomial_degree, nt, T, periodic)
    return np.array(points)


def get_langrange_points(
    polynomial_degree: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    """Backward-compatible alias for the historically misspelled name."""
    return get_lagrange_points(
        polynomial_degree, nt, T, periodic=periodic
    )



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

def get_lagrange_prolongation_matrix(
    nt_coarse: int,
    nt_fine: int,
    polynomial_degree: int,
    periodic: bool = False,
):
    """Get the prolongation matrix from coarse to fine grid in Lagrange basis."""
    if periodic and polynomial_degree < 1:
        raise ValueError("periodic Lagrange spaces require polynomial_degree >= 1")
    if periodic and (nt_coarse < 1 or nt_fine < 1):
        raise ValueError("periodic Lagrange prolongation requires positive interval counts")

    from hmod.hmod import lagrange_prolongation_matrix
    from scipy.sparse import coo_matrix
    row_indices, col_indices, values = lagrange_prolongation_matrix(
        nt_coarse, nt_fine, polynomial_degree, periodic
    )
    endpoint_dofs = 0 if periodic else 1
    shape = (
        nt_fine * polynomial_degree + endpoint_dofs,
        nt_coarse * polynomial_degree + endpoint_dofs,
    )
    return coo_matrix((values, (row_indices, col_indices)), shape=shape).tocsr()


def get_legendre_derivative_matrix(
    nt: int,
    p: int,
    T: float,
    square: bool = True,
    periodic: bool = False,
):
    """
    Global sparse matrix for d/dt on piecewise-Legendre coefficients
    with degree-major ordering:
        [u_0(all intervals), u_1(all intervals), ..., u_p(all intervals)]^T

    Uses the standard Legendre basis P_n on each interval.

    Parameters
    ----------
    nt : int
        Number of equal intervals.
    p : int
        Polynomial degree on each interval.
    T : float
        Total time interval length.
    square : bool
        If False, return the natural rectangular map into degree <= p-1.
        If True, append one zero block-row to make the matrix square.
    periodic : bool
        Accepted for API consistency. Differentiation in the discontinuous
        local Legendre representation is the same for both topologies.
    """
    nrows_deg = p + 1 if square else p
    ncols_deg = p + 1

    blocks = [[None for _ in range(ncols_deg)] for _ in range(nrows_deg)]

    h = T / nt
    scale = 2.0 / h
    I = sp.eye(nt, format="csr")

    for n in range(ncols_deg):          # input degree
        blocks[n][n] = 0. * I #to be sure that we have at least one matrix in each row
        for m in range(min(n, nrows_deg)):   # output degree
            if (n - m) % 2 == 1:
                blocks[m][n] = scale * (2*m + 1) * I

    return sp.bmat(blocks, format="csr")









