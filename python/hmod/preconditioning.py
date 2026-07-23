import scipy.sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np


class GMRESCounter:
    """Callable iteration counter for ``scipy.sparse.linalg.gmres``.

    The callback argument is stored as the current residual. With SciPy's
    default GMRES callback behavior this is the preconditioned residual norm.
    """

    def __init__(self, print_residual: bool = False):
        self.print_residual = print_residual
        self.niter = 0
        self.residuals = []

    def __call__(self, residual):
        self.niter += 1
        self.residuals.append(residual)
        if self.print_residual:
            print(f"GMRES iteration {self.niter}: residual = {residual}")

    @property
    def current_residual(self):
        """Return the most recently reported residual, if available."""
        if not self.residuals:
            return None
        return self.residuals[-1]


class LU_solver(scipy.sparse.linalg.LinearOperator):
    """LinearOperator wrapper around a sparse LU factorization."""

    def __init__(self, A : scipy.sparse.spmatrix):
        self.A_lu = scipy.sparse.linalg.splu(A.tocsc())
        self.dtype = np.float64
        shape = A.shape
        super().__init__(dtype=self.dtype, shape=shape)
    def _matmat(self, X):
        """Solve the factored sparse system for one or more right-hand sides."""
        return self.A_lu.solve(X)

class BPXPreconditioner(LinearOperator):
    r"""BPX preconditioner for temporal Lagrange spaces.

    This operator approximates the inverse of the temporal norm operator

    ``mu * I + B^s``,

    where ``B^s`` represents a Sobolev-type contribution of order
    ``sobolev_exponent``. In the hybrid ODE setting one typically uses
    ``sobolev_exponent=0.5`` to precondition the
    ``H^{1/2}(I) + mu L2(I)`` norm induced by the formulation.

    The implementation builds a hierarchy of uniformly refined temporal
    meshes from ``nt_coarse`` to ``nt_coarse * 2**n_refinements``. On each
    level it forms the Lagrange mass matrix ``M_l``, weights it by
    ``mu + h_l**(-2*s)``, applies a sparse LU inverse on that level, and
    prolongates the result to the finest level. The level contributions are
    summed in the standard BPX fashion.

    By default, the first temporal DOF is removed on every level, which
    corresponds to a homogeneous initial condition such as ``u(0)=0``.
    With ``periodic=True``, endpoint identification is built into the
    Lagrange space and no DOF is removed.

    Parameters
    ----------
    mu:
        Weight of the L2/mass part.
    n_refinements:
        Number of uniform refinements from the coarsest to finest mesh.
    sobolev_exponent:
        Sobolev order ``s`` used in the BPX level weight ``h_l**(-2*s)``.
    polynomial_degree:
        Polynomial degree of the temporal Lagrange space on each interval.
    nt_coarse:
        Number of time intervals on the coarsest mesh.
    T:
        Final time, i.e. the interval is ``(0, T)``.
    periodic:
        Use periodic Lagrange spaces and periodic prolongation.
    """
    def __init__(
        self,
        mu: float,
        n_refinements: int,
        sobolev_exponent: float,
        polynomial_degree: int,
        nt_coarse: int,
        T: float,
        periodic: bool = False,
    ):
        from hmod.standard_matrices import get_lagrange_lagrange_matrix_for_derivatives
        import hmod.polynomial_bases as pb
        nt_finest = nt_coarse * (2**n_refinements)
        level_ops = []
        for i in range(n_refinements+1):
            nt = nt_coarse * (2**i)
            h = T/nt
            M = get_lagrange_lagrange_matrix_for_derivatives(
                polynomial_degree,
                polynomial_degree,
                0,
                0,
                nt,
                T,
                periodic=periodic,
            )
            P = pb.get_lagrange_prolongation_matrix(
                nt,
                nt_finest,
                polynomial_degree,
                periodic=periodic,
            )
            if not periodic:
                # Homogenize the initial condition in the non-periodic space.
                M = M[1:, :][:, 1:]
                P = P[1:, :][:, 1:]
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
        """Apply the BPX preconditioner to the active finest-level vector."""
        x = np.array(x)
        x2 = self.B_inv_op @ x
        return x2
