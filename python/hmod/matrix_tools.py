from scipy.sparse.linalg import LinearOperator
import numpy as  np
from scipy.sparse import csr_matrix
from numba import njit

def linear_operator_to_matrix(L):
        #this step is costly now, but can be replaced later by a more efficient implementation
        n_cols = L.shape[1]
        A = np.zeros(L.shape)
        e = np.zeros(n_cols)
        for i in range(n_cols):
            e[i] = 1.0
            A[:, i] = L @ e
            e[i] = 0.0
        return A

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