import scipy.sparse
from fontTools.misc.bezierTools import Identity
from scipy.sparse.linalg import LinearOperator
import numpy as np
#import pypardiso as ppd

class LowestOrderPreconditioner(LinearOperator):
    """A linear operator that applies a lowest-order preconditioner."""
    def __init__(self, nt : int, mu : float, T : float, n_terms :int = int(1e5)):
        from hmod.hmod import EigenBasisTransformLowestOrder
        import scipy.sparse as sp
        import hmod.standard_matrices as sm
        self.nt = nt
        self.Tmax = T
        self.n_terms = n_terms
        self.eigen_basis_transform = EigenBasisTransformLowestOrder(nt, T)
        generalized_eigenvalues = self.eigen_basis_transform.get_generalized_eigenvalues(n_terms)
        generalized_eigenvalues = np.array(generalized_eigenvalues)
        #get diagonal matrix with eigenvalues
        inverse_diagonal = 1.0/(generalized_eigenvalues+mu)
        self.inverse_in_spectrum = sp.diags(inverse_diagonal)
        #get p.w. linear mass matrix
        self.mass = sm.get_lagrange_lagrange_matrix_for_derivatives(1,1,0,0,nt,T)
        #get rid of first row and column to account for homogeneous BCs
        self.mass = self.mass[1:,1:]
        self.mass_lu = scipy.sparse.linalg.splu(self.mass)
        self.dtype = np.float64
        n_dofs = nt
        shape = (n_dofs, n_dofs)
        super().__init__(dtype=self.dtype, shape=shape)

    def _matvec(self, x):
        """Apply the lowest-order preconditioner to the input vector x."""
        # first apply the inverse mass using pypardiso
        #f1 = ppd.spsolve(self.mass, x)
        f1 = self.mass_lu.solve(x)
        #then inverse transform
        f2 = self.eigen_basis_transform.inverse_transform(f1)
        #then apply the inverse in the spectrum
        v = self.inverse_in_spectrum @ f2
        #then forward transform
        u = self.eigen_basis_transform.forward_transform(v)
        return u



class GeneralOrderPreconditioner(LinearOperator):
    """A linear operator that applies a general-order preconditioner.
       At the moment, we just apply the lowest order preconditioner to its respective subspace and leave the rest unchanged.
    """
    def __init__(self, polynomial_order : int, nt : int, mu : float, T : float, n_terms :int = int(1e5)):
        import scipy.sparse as sp
        self.lowest_order_preconditioner = LowestOrderPreconditioner(nt, mu, T, n_terms)
        self.dtype = np.float64
        n_dofs = polynomial_order*nt
        self.lowest_order_dofs = np.arange(1,nt+1)*polynomial_order-1  #excluding first dof for homogeneous BCs
        shape = (n_dofs, n_dofs)
        super().__init__(dtype=self.dtype, shape=shape)

    def _matvec(self, x):
        """Apply the general-order preconditioner to the input vector x."""
        #extract the lowest order dofs
        x_lowest = x[self.lowest_order_dofs]
        #apply the lowest order preconditioner
        u_lowest = self.lowest_order_preconditioner @ x_lowest
        #insert back
        u = np.copy(x)
        u[self.lowest_order_dofs] = u_lowest
        return u



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