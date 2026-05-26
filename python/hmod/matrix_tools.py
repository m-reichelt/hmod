from scipy.sparse.linalg import LinearOperator
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit

def linear_operator_to_matrix(L):
        """Materialize a linear operator as a dense matrix.

        This also works for rectangular operators with shape ``(m, n)``:
        applying ``L`` to the ``n x n`` identity returns the full ``m x n``
        dense matrix.
        """
        # Apply the operator to all basis vectors at once, so LinearOperator
        # implementations can use their optimized/parallel matmat path.
        n_cols = L.shape[1]
        identity = np.eye(n_cols, dtype=getattr(L, "dtype", np.float64))
        return np.asarray(L @ identity)

def triplets_to_linear_operator(row_indices, col_indices, values):
    from scipy.sparse import coo_matrix
    sparse_matrix = coo_matrix((values, (row_indices, col_indices)))
    return sparse_matrix


@njit
def csr_matvec_reverse_numba(indptr, indices, data, x):
    n_rows = indptr.size - 1
    y = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i+1]
        s = 0.0
        # walk the row backwards
        for j in range(end - 1, start - 1, -1):
            s += data[j] * x[indices[j]]
        y[i] = s
    return y


def csr_matvec_reverse(A: csr_matrix, x: np.ndarray) -> np.ndarray:
    indptr = A.indptr
    indices = A.indices
    data = A.data
    return csr_matvec_reverse_numba(indptr, indices, data, x)
