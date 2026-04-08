import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import aslinearoperator

from hmod.block_operations import BlockLinearOperator


def test_block_linear_operator_against_sparse_block_matrix():
    A_mat = np.array([[1.0, 2.0],
                      [3.0, 4.0]])
    B_mat = np.array([[5.0],
                      [6.0]])
    C_mat = np.array([[7.0, 8.0]])

    A = aslinearoperator(A_mat)
    B = aslinearoperator(B_mat)
    C = aslinearoperator(C_mat)

    Lop = BlockLinearOperator([
        [A, B],
        [C, None],
    ])

    Lmat = bmat([
        [csr_matrix(A_mat), csr_matrix(B_mat)],
        [csr_matrix(C_mat), None],
    ], format="csr")

    x = np.array([1.0, -1.0, 2.0])
    X = np.array([[1.0, 0.0],
                  [-1.0, 2.0],
                  [2.0, 3.0]])

    y_op = Lop @ x
    y_mat = Lmat @ x
    np.testing.assert_allclose(y_op, y_mat, rtol=1e-13, atol=1e-13)

    Y_op = Lop @ X
    Y_mat = Lmat @ X
    np.testing.assert_allclose(Y_op, Y_mat, rtol=1e-13, atol=1e-13)

    z = np.array([1.0, 3.0, 5.0])
    Z = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])

    yt_op = Lop.H @ z
    yt_mat = Lmat.T.conj() @ z
    np.testing.assert_allclose(yt_op, yt_mat, rtol=1e-13, atol=1e-13)

    Yt_op = Lop.H @ Z
    Yt_mat = Lmat.T.conj() @ Z
    np.testing.assert_allclose(Yt_op, Yt_mat, rtol=1e-13, atol=1e-13)