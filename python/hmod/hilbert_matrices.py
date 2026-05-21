import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np
import hmod.block_operations as block_ops
import hmod.standard_matrices as sm
import hmod.polynomial_bases as pb
from scipy.special import factorial, zeta


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
        return self._matmat(x)  # dst-iv is self-adjoint


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
        return self._matmat(x)  # dct-iv is self-adjoint


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


def a_k_12_fun(k: int, m: int):
    if k > m:
        return 0.
    else:
        numerator = factorial(m + k)
        denominator = factorial(k) * factorial(m - k) * 2 ** k
        return numerator / denominator


def first_branch_sum(diagonal_index: int, nt: int, pol_deg_trial: int, pol_deg_test: int):
    from scipy.special import spherical_jn as jn
    # use shorthand notation for the indices as in paper
    r = pol_deg_test
    m = pol_deg_trial
    a = diagonal_index
    # define lambdas
    a_m_i = lambda m, i: (-1) ** i * a_k_12_fun(2 * i, m)
    b_m_i = lambda m, i: (-1) ** i * a_k_12_fun(2 * i + 1, m)
    beta_a = np.pi * (2 * a + 1) / (4 * nt)
    sigma_m_a = lambda m: np.sin(beta_a - m * np.pi / 2)
    kappa_m_a = lambda m: np.cos(beta_a - m * np.pi / 2)
    q_shift = 1. + beta_a / np.pi
    sum_val = jn(r, beta_a) * jn(m, beta_a)
    for i in range(0, r // 2 + 1):
        for j in range(0, m // 2 + 1):
            fac = a_m_i(r, i) * a_m_i(m, j) * sigma_m_a(r) * sigma_m_a(m)
            arg = 2 * (i + j) + 2
            zeta_val = zeta(arg, q_shift) / np.pi ** arg
            sum_val += fac * zeta_val
    for i in range(0, r // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            fac = a_m_i(r, i) * b_m_i(m, j) * sigma_m_a(r) * kappa_m_a(m)
            arg = 2 * (i + j) + 3
            zeta_val = zeta(arg, q_shift) / np.pi ** arg
            sum_val += fac * zeta_val
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, m // 2 + 1):
            fac = b_m_i(r, i) * a_m_i(m, j) * kappa_m_a(r) * sigma_m_a(m)
            arg = 2 * (i + j) + 3
            zeta_val = zeta(arg, q_shift) / np.pi ** arg
            sum_val += fac * zeta_val
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            fac = b_m_i(r, i) * b_m_i(m, j) * kappa_m_a(r) * kappa_m_a(m)
            arg = 2 * (i + j) + 4
            zeta_val = zeta(arg, q_shift) / np.pi ** arg
            sum_val += fac * zeta_val

    return sum_val / nt ** 2


def second_branch_sum(diagonal_index: int, nt: int, pol_deg_trial: int, pol_deg_test: int):
    # use shorthand notation for the indices as in paper
    r = pol_deg_test
    m = pol_deg_trial
    a = diagonal_index
    # define lambdas
    a_m_i = lambda m, i: (-1) ** i * a_k_12_fun(2 * i, m)
    b_m_i = lambda m, i: (-1) ** i * a_k_12_fun(2 * i + 1, m)
    beta_a = np.pi * (2 * a + 1) / (4 * nt)
    sigma_m_a = lambda m: np.sin(beta_a - m * np.pi / 2)
    kappa_m_a = lambda m: np.cos(beta_a - m * np.pi / 2)
    sum_val = 0.
    for i in range(0, r // 2 + 1):
        for j in range(0, m // 2 + 1):
            fac = a_m_i(r, i) * a_m_i(m, j) * sigma_m_a(r) * sigma_m_a(m)
            arg = 2 * (i + j) + 2
            zeta_val = zeta(arg, 1. - beta_a / np.pi) / np.pi ** arg
            sum_val += fac * zeta_val * (-1) ** (r + m)
    for i in range(0, r // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            fac = a_m_i(r, i) * b_m_i(m, j) * sigma_m_a(r) * kappa_m_a(m)
            arg = 2 * (i + j) + 3
            zeta_val = zeta(arg, 1. - beta_a / np.pi) / np.pi ** arg
            sum_val += fac * zeta_val * (-1) ** (r + m + 1)
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, m // 2 + 1):
            fac = b_m_i(r, i) * a_m_i(m, j) * kappa_m_a(r) * sigma_m_a(m)
            arg = 2 * (i + j) + 3
            zeta_val = zeta(arg, 1. - beta_a / np.pi) / np.pi ** arg
            sum_val += fac * zeta_val * (-1) ** (r + m + 1)
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            fac = b_m_i(r, i) * b_m_i(m, j) * kappa_m_a(r) * kappa_m_a(m)
            arg = 2 * (i + j) + 4
            zeta_val = zeta(arg, 1. - beta_a / np.pi) / np.pi ** arg
            sum_val += fac * zeta_val * (-1) ** (r + m)

    return sum_val / nt ** 2


def get_kernel_matrix_for_degrees_zeta(nt: int, pol_deg_trial: int, pol_deg_test: int):
    fac = -1.
    if pol_deg_trial % 2 != pol_deg_test % 2:
        fac = 1.
    entry_i = lambda i: first_branch_sum(i, nt, pol_deg_trial,
                                         pol_deg_test) + fac * second_branch_sum(i, nt,
                                                                                 pol_deg_trial,
                                                                                 pol_deg_test)
    return sparse.diags([entry_i(i) for i in range(nt)], format='csr')


def get_kernel_matrix(nt: int, pol_deg_trial: int, pol_deg_test: int):
    blocks = [[get_kernel_matrix_for_degrees_zeta(nt, n, m) for n in range(pol_deg_trial + 1)]
              for m in
              range(pol_deg_test + 1)]
    return sparse.bmat(blocks, format='csr')


def get_operator_I_H_legendre_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int, final_time: float):
    U = get_trial_transform(pol_deg_trial, nt)
    T = get_test_transform(pol_deg_test, nt)
    K = sparse.linalg.aslinearoperator(get_kernel_matrix(nt, pol_deg_trial, pol_deg_test))
    return (final_time * 0.5) * T.H @ K @ U


def get_hilbert_matrix_with_derivatives_legendre_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                          derivatives_trial: int, derivatives_test: int,
                                                          final_time: float):
    Mh = get_operator_I_H_legendre_legendre(nt, pol_deg_trial, pol_deg_test, final_time)
    Mat = Mh
    if derivatives_trial > 0:
        D_trial = pb.get_legendre_derivative_matrix(nt, pol_deg_trial, final_time)
        D_pow = sparse.linalg.matrix_power(D_trial, derivatives_trial)
        D_op = sparse.linalg.aslinearoperator(D_pow)
        Mat = Mat @ D_op
    if derivatives_test > 0:
        D_test = pb.get_legendre_derivative_matrix(nt, pol_deg_test, final_time)
        D_pow = sparse.linalg.matrix_power(D_test, derivatives_test)
        D_op = sparse.linalg.aslinearoperator(D_pow)
        Mat = D_op.H @ Mat

    return Mat


def get_hilbert_matrix_with_derivatives_legendre_lagrange(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                          derivatives_trial: int, derivatives_test: int,
                                                          final_time: float):
    K = get_hilbert_matrix_with_derivatives_legendre_legendre(nt, pol_deg_trial
                                                              , pol_deg_test, derivatives_trial, derivatives_test,
                                                              final_time)
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_test, nt))

    return trans.T @ K


def get_hilbert_matrix_with_derivatives_lagrange_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                          derivatives_trial: int, derivatives_test: int,
                                                          final_time: float):
    K = get_hilbert_matrix_with_derivatives_legendre_legendre(nt, pol_deg_trial
                                                              , pol_deg_test, derivatives_trial, derivatives_test,
                                                              final_time)
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_trial, nt))

    return K @ trans


def get_hilbert_matrix_with_derivatives_lagrange_lagrange(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                          derivatives_trial: int, derivatives_test: int,
                                                          final_time: float):
    K = get_hilbert_matrix_with_derivatives_legendre_legendre(nt, pol_deg_trial
                                                              , pol_deg_test, derivatives_trial, derivatives_test,
                                                              final_time)
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_trial, nt))
    transT = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_test, nt))

    return transT.T @ K @ trans
