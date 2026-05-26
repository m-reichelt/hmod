from scipy.sparse.linalg import LinearOperator
import numpy as np

class DofRestrictorSymmetric(LinearOperator):
    """Restrict a square operator to its active degrees of freedom.

    This wrapper is useful for imposing essential conditions such as the
    initial value ``u(0) = 0`` in temporal systems. It exposes the reduced
    operator on all non-restricted DOFs. During a matrix-vector product, the
    input vector is embedded into the full vector, the restricted DOFs are
    filled with the prescribed values, the unrestricted operator is applied,
    and only the active rows are returned.

    For an unrestricted square operator ``A`` and a full vector split into
    active and restricted DOFs, this corresponds to applying the active block
    of ``A`` while accounting for fixed restricted values. For homogeneous
    values, this is the usual row/column restriction.

    Parameters
    ----------
    unrestricted_operator:
        Square operator acting on the full set of DOFs.
    restricted_dofs:
        Indices of the DOFs to remove from the reduced operator.
    restricted_dof_values:
        Values inserted at ``restricted_dofs`` before applying the full
        operator.
    """
    def __init__(self, unrestricted_operator : LinearOperator
                 , restricted_dofs : np.ndarray, restricted_dof_values : np.ndarray):
        self.unrestricted_operator = unrestricted_operator
        self.restricted_dofs = restricted_dofs
        self.restricted_dof_values = restricted_dof_values
        self.dtype = np.float64  # or any other appropriate dtype
        n_restricted_dofs = len(restricted_dofs)
        n_unrestricted_dofs = unrestricted_operator.shape[0]
        n_dofs = n_unrestricted_dofs - n_restricted_dofs
        #get bit array with active dofs
        self.active_dofs = np.ones(n_unrestricted_dofs, dtype=bool)
        self.active_dofs[restricted_dofs] = False
        #and the negation for setting values
        self.inactive_dofs = ~self.active_dofs
        shape = (n_dofs, n_dofs)
        super().__init__(dtype=self.dtype, shape=shape)

    def _matvec(self, x):
        """Apply the restricted operator to a reduced vector."""
        x_full = np.zeros(self.unrestricted_operator.shape[0])
        x_full[self.active_dofs] = x.ravel()
        x_full[self.inactive_dofs] = self.restricted_dof_values
        y_full = self.unrestricted_operator @ x_full
        y_restricted = y_full[self.active_dofs]
        return y_restricted
    
    def _matmat(self, X):
        """Apply the restricted operator to multiple reduced vectors."""
        #do this explicitly in sequential way to avoid shape issues
        ncols = X.shape[1]
        result = np.zeros((self.shape[0], ncols))
        for i in range(ncols):
            result[:, i] = self._matvec(X[:, i])
        return result
