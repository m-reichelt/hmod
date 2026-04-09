import scipy.sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np



class LU_solver(scipy.sparse.linalg.LinearOperator):
    def __init__(self, A : scipy.sparse.spmatrix):
        self.A_lu = scipy.sparse.linalg.splu(A.tocsc())
        self.dtype = np.float64
        shape = A.shape
        super().__init__(dtype=self.dtype, shape=shape)
    def _matmat(self, X):
        return self.A_lu.solve(X)

class BPXPreconditioner(LinearOperator):
    """A linear operator that applies the BPX preconditioner."""
    def __init__(self, mu : float, n_refinements : int, sobolev_exponent : float, polynomial_degree : int, nt_coarse : int, T : float):
        from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives
        import hmod.polynomial_bases as pb
        nt_finest = nt_coarse * (2**n_refinements)
        level_ops = []
        for i in range(n_refinements+1):
            nt = nt_coarse * (2**i)
            h = T/nt
            M = get_lagrange_lagrange_matrix_for_derivatives(polynomial_degree, polynomial_degree, 0, 0, nt, T)
            P = pb.get_lagrange_prolongation_matrix(nt, nt_finest, polynomial_degree)
            #homogenize initial conditions
            M = M[1:,:][:,1:]
            P = P[1:,:][:,1:]
            M_weighted = M *(mu + h**(-2.0 * sobolev_exponent))
            M_weighted_inverse_op = LU_solver(M_weighted)
            P = scipy.sparse.linalg.aslinearoperator(P)
            level_op = P @ M_weighted_inverse_op @ P.transpose()
            level_ops.append(level_op)
        #sum all level operators
        self.B_inv_op = level_ops[0]
        for op in level_ops[1:]:
            self.B_inv_op = self.B_inv_op + op

        self.dtype = np.float64
        shape = self.B_inv_op.shape
        super().__init__(dtype=self.dtype, shape=shape)

    def _matvec(self, x):
        """Apply the BPX preconditioner to the input vector x."""
        x = np.array(x)
        x2 = self.B_inv_op @ x
        return x2