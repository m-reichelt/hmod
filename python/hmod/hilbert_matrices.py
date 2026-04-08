import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np
import hmod.block_operations as block_ops


class DST_IV(LinearOperator):
    def __init__(self, nt: int, workers=-1):
        self.nt = nt
        self.workers = workers
        shape = (nt, nt)
        super().__init__(dtype=np.float64, shape=shape)

    def _matmat(self, x):
        from scipy.fft import dst
        return dst(x, type=4, axis=0, workers=self.workers)

    def _rmatmat(self, x):
        return self._matmat(x)  #dst-iv is self-adjoint


class DCT_IV(LinearOperator):
    def __init__(self, nt: int, workers=-1):
        self.nt = nt
        self.workers = workers
        shape = (nt, nt)
        super().__init__(dtype=np.float64, shape=shape)

    def _matmat(self, x):
        from scipy.fft import dct
        return dct(x, type=4, axis=0, workers=self.workers)

    def _rmatmat(self, x):
        return self._matmat(x)  #dct-iv is self-adjoint


def get_trial_transform(pol_deg, nt, workers=-1):
    from hmod.block_operations import BlockLinearOperator
    blocks = [[None for _ in range(pol_deg + 1)] for _ in range(pol_deg + 1)]

    for m in range(pol_deg + 1):
        if m % 2 == 0:
            fac = np.power(-1, m // 2)
            blocks[m][m] = fac * DCT_IV(nt, workers=workers)
        else:
            fac = np.power(-1, (m + 1) // 2)
            blocks[m][m] = fac * DST_IV(nt, workers=workers)

    block_operator = block_ops.BlockLinearOperator(blocks)
    return block_operator

def get_test_transform(pol_deg, nt, workers=-1):
    from hmod.block_operations import BlockLinearOperator
    blocks = [[None for _ in range(pol_deg + 1)] for _ in range(pol_deg + 1)]

    for m in range(pol_deg + 1):
        if m % 2 == 0:
            fac = np.power(-1, m // 2)
            blocks[m][m] = fac * DST_IV(nt, workers=workers)
        else:
            fac = np.power(-1, (m - 1) // 2)
            blocks[m][m] = fac * DCT_IV(nt, workers=workers)

    block_operator = block_ops.BlockLinearOperator(blocks)
    return block_operator

def first_branch_sum(diagonal_index : int, nt : int, pol_deg_trial : int, pol_deg_test : int, workers : int = -1):
    assert 0 <= diagonal_index < nt, "Diagonal index must be between 0 and nt-1"
    from scipy.special import spherical_jn as jn
    alpha_k = lambda k : np.pi*(2*k+1)/(4*nt)
    alpha_q = lambda q : alpha_k(2*q*nt+diagonal_index)
    nmodes = int(1e5)
    summand = lambda q : jn(pol_deg_test, alpha_q(q))*jn(pol_deg_trial, alpha_q(q))
    #sum in reversed order
    q_vals = np.arange(nmodes)[::-1]
    return np.sum(summand(q_vals))/(nt**2)

def second_branch_sum(diagonal_index : int, nt : int, pol_deg_trial : int, pol_deg_test : int, workers : int = -1):
    assert 0 <= diagonal_index < nt, "Diagonal index must be between 0 and nt-1"
    from scipy.special import spherical_jn as jn
    alpha_k = lambda k : np.pi*(2*k+1)/(4*nt)
    alpha_q = lambda q : alpha_k(2*nt*(q+1)-1-diagonal_index)
    nmodes = int(1e5)
    summand = lambda q : jn(pol_deg_test, alpha_q(q))*jn(pol_deg_trial, alpha_q(q))
    #sum in reversed order
    q_vals = np.arange(nmodes)[::-1]
    return np.sum(summand(q_vals))/(nt**2)


def get_kernel_matrix_for_degrees(nt : int, pol_deg_trial : int, pol_deg_test : int):
    fac = -1.
    if pol_deg_trial % 2 != pol_deg_test % 2:
        fac = 1.
    entry_i = lambda i : first_branch_sum(i, nt, pol_deg_trial, pol_deg_test) + fac*second_branch_sum(i, nt, pol_deg_trial, pol_deg_test)
    return sparse.diags([entry_i(i) for i in range(nt)], format='csr')


def get_kernel_matrix(nt : int, pol_deg_trial : int, pol_deg_test : int):
    blocks = [[get_kernel_matrix_for_degrees(nt, n, m) for n in range(pol_deg_trial + 1)] for m in range(pol_deg_test + 1)]
    return sparse.bmat(blocks, format='csr')