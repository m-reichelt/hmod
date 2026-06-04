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
        self.restricted_dofs = np.asarray(restricted_dofs, dtype=int).ravel()
        self.restricted_dof_values = np.asarray(restricted_dof_values).ravel()
        self.dtype = np.float64  # or any other appropriate dtype
        n_restricted_dofs = len(self.restricted_dofs)
        if self.restricted_dof_values.size != n_restricted_dofs:
            raise ValueError(
                "restricted_dof_values must have the same length as restricted_dofs "
                f"({self.restricted_dof_values.size} != {n_restricted_dofs})"
            )
        n_unrestricted_dofs = unrestricted_operator.shape[0]
        n_dofs = n_unrestricted_dofs - n_restricted_dofs
        #get bit array with active dofs
        self.active_dofs = np.ones(n_unrestricted_dofs, dtype=bool)
        self.active_dofs[self.restricted_dofs] = False
        #and the negation for setting values
        self.inactive_dofs = ~self.active_dofs
        shape = (n_dofs, n_dofs)
        super().__init__(dtype=self.dtype, shape=shape)

    def _matmat(self, X):
        """Apply the restricted operator to multiple reduced vectors."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"_matmat expects a 2D array, got shape {X.shape!r}")
        if X.shape[0] != self.shape[1]:
            raise ValueError(
                f"dimension mismatch in _matmat: operator shape {self.shape}, "
                f"got X with shape {X.shape}"
            )

        n_full = self.unrestricted_operator.shape[0]
        ncols = X.shape[1]
        dtype = np.result_type(self.dtype, X.dtype, self.restricted_dof_values.dtype)

        X_full = np.empty((n_full, ncols), dtype=dtype)
        X_full[self.active_dofs, :] = X
        X_full[self.restricted_dofs, :] = self.restricted_dof_values[:, None]

        Y_full = self.unrestricted_operator @ X_full
        return np.asarray(Y_full)[self.active_dofs, :]
