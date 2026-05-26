import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np
import hmod.block_operations as block_ops
import hmod.standard_matrices as sm
import hmod.polynomial_bases as pb
from scipy.special import factorial, spherical_jn, zeta


class DST_IV(LinearOperator):
    """Matrix-free discrete sine transform of type IV on nt values."""

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
    """Matrix-free discrete cosine transform of type IV on nt values."""

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
    """Return the FFT-based block transform for trial Legendre modes.

    The transform maps degree-major piecewise Legendre coefficients into
    the sine/cosine coefficient ordering used by the modified Hilbert
    transform kernel. Even trial degrees use DCT-IV blocks, odd trial
    degrees use DST-IV blocks, with the signs from the analytic formula.
    """
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
    """Return the FFT-based block transform for test Legendre modes.

    This is the test-side analogue of :func:`get_trial_transform`. It maps
    piecewise Legendre test coefficients into the sine/cosine coefficient
    ordering used for the action of ``H_T`` on the test function in
    ``<u, H_T v>_I``.
    """
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
    """Auxiliary coefficient in the zeta-series kernel formula."""
    if k > m:
        return 0.
    else:
        numerator = factorial(m + k)
        denominator = factorial(k) * factorial(m - k) * 2 ** k
        return numerator / denominator


def first_branch_sum(diagonal_index: int, nt: int, pol_deg_trial: int, pol_deg_test: int):
    """Compute the first zeta-series branch for one kernel diagonal entry.

    The returned scalar contributes to the diagonal kernel block coupling
    trial Legendre degree ``pol_deg_trial`` with test Legendre degree
    ``pol_deg_test`` for the bilinear form ``<u, H_T v>_I``.
    """
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
    """Compute the second zeta-series branch for one kernel diagonal entry.

    Together with :func:`first_branch_sum`, this gives the diagonal kernel
    block for a fixed pair of trial/test Legendre degrees in the matrix of
    ``<u, H_T v>_I``.
    """
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


def _a_m_i(m: int, i: int):
    return (-1) ** i * a_k_12_fun(2 * i, m)


def _b_m_i(m: int, i: int):
    return (-1) ** i * a_k_12_fun(2 * i + 1, m)


def _kernel_diagonal_entries_zeta(nt: int, pol_deg_trial: int, pol_deg_test: int):
    """Vectorized diagonal entries for one transformed Hilbert kernel block."""
    r = pol_deg_test
    m = pol_deg_trial

    diagonal_indices = np.arange(nt, dtype=np.float64)
    beta = np.pi * (2 * diagonal_indices + 1) / (4 * nt)
    q_plus = 1.0 + beta / np.pi
    q_minus = 1.0 - beta / np.pi

    sigma_r = np.sin(beta - r * np.pi / 2)
    sigma_m = np.sin(beta - m * np.pi / 2)
    kappa_r = np.cos(beta - r * np.pi / 2)
    kappa_m = np.cos(beta - m * np.pi / 2)

    first = spherical_jn(r, beta) * spherical_jn(m, beta)
    for i in range(0, r // 2 + 1):
        for j in range(0, m // 2 + 1):
            arg = 2 * (i + j) + 2
            first += (
                _a_m_i(r, i)
                * _a_m_i(m, j)
                * sigma_r
                * sigma_m
                * zeta(arg, q_plus)
                / np.pi ** arg
            )
    for i in range(0, r // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            arg = 2 * (i + j) + 3
            first += (
                _a_m_i(r, i)
                * _b_m_i(m, j)
                * sigma_r
                * kappa_m
                * zeta(arg, q_plus)
                / np.pi ** arg
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, m // 2 + 1):
            arg = 2 * (i + j) + 3
            first += (
                _b_m_i(r, i)
                * _a_m_i(m, j)
                * kappa_r
                * sigma_m
                * zeta(arg, q_plus)
                / np.pi ** arg
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            arg = 2 * (i + j) + 4
            first += (
                _b_m_i(r, i)
                * _b_m_i(m, j)
                * kappa_r
                * kappa_m
                * zeta(arg, q_plus)
                / np.pi ** arg
            )

    second = np.zeros(nt, dtype=np.float64)
    for i in range(0, r // 2 + 1):
        for j in range(0, m // 2 + 1):
            arg = 2 * (i + j) + 2
            second += (
                _a_m_i(r, i)
                * _a_m_i(m, j)
                * sigma_r
                * sigma_m
                * zeta(arg, q_minus)
                * (-1) ** (r + m)
                / np.pi ** arg
            )
    for i in range(0, r // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            arg = 2 * (i + j) + 3
            second += (
                _a_m_i(r, i)
                * _b_m_i(m, j)
                * sigma_r
                * kappa_m
                * zeta(arg, q_minus)
                * (-1) ** (r + m + 1)
                / np.pi ** arg
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, m // 2 + 1):
            arg = 2 * (i + j) + 3
            second += (
                _b_m_i(r, i)
                * _a_m_i(m, j)
                * kappa_r
                * sigma_m
                * zeta(arg, q_minus)
                * (-1) ** (r + m + 1)
                / np.pi ** arg
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            arg = 2 * (i + j) + 4
            second += (
                _b_m_i(r, i)
                * _b_m_i(m, j)
                * kappa_r
                * kappa_m
                * zeta(arg, q_minus)
                * (-1) ** (r + m)
                / np.pi ** arg
            )

    second_branch_sign = -1.0
    if pol_deg_trial % 2 != pol_deg_test % 2:
        second_branch_sign = 1.0

    return (first + second_branch_sign * second) / nt ** 2


def get_kernel_matrix_for_degrees_zeta(nt: int, pol_deg_trial: int, pol_deg_test: int):
    """Return one diagonal kernel block for fixed Legendre degrees.

    The block has shape ``(nt, nt)`` and couples all time intervals for one
    trial degree and one test degree. It is diagonal in the transformed
    sine/cosine basis; the entries are evaluated with the zeta-series
    formula used for the modified Hilbert transform.
    """
    entries = _kernel_diagonal_entries_zeta(nt, pol_deg_trial, pol_deg_test)
    return sparse.diags(entries, format='csr')


def get_kernel_matrix(nt: int, pol_deg_trial: int, pol_deg_test: int):
    """Return the full transformed kernel for all Legendre degree pairs.

    The result is a block matrix collecting
    :func:`get_kernel_matrix_for_degrees_zeta` for trial degrees
    ``0..pol_deg_trial`` and test degrees ``0..pol_deg_test``.
    """
    blocks = [[get_kernel_matrix_for_degrees_zeta(nt, n, m) for n in range(pol_deg_trial + 1)]
              for m in
              range(pol_deg_test + 1)]
    return sparse.bmat(blocks, format='csr')


def get_operator_I_H_legendre_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int, final_time: float):
    r"""Return the Legendre-Legendre operator for ``<u, H_T v>_I``.

    Rows correspond to test functions and columns to trial functions. For
    trial basis functions ``phi_m`` and test basis functions ``psi_r``,
    this operator represents

    ``A[r, m] = <phi_m, H_T psi_r>_I``.

    The returned object is matrix-free and uses FFT-based transforms around
    the zeta-series kernel.
    """
    U = get_trial_transform(pol_deg_trial, nt)
    T = get_test_transform(pol_deg_test, nt)
    K = sparse.linalg.aslinearoperator(get_kernel_matrix(nt, pol_deg_trial, pol_deg_test))
    return (final_time * 0.5) * T.H @ K @ U


def get_hilbert_matrix_for_derivatives_legendre_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                         derivatives_trial: int, derivatives_test: int,
                                                         final_time: float):
    r"""Return a Hilbert matrix in Legendre trial/test bases.

    Rows correspond to test basis functions and columns to trial basis
    functions. With ``i = derivatives_trial`` and
    ``j = derivatives_test``, the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``.

    Here ``phi_m`` is a trial basis function, ``psi_r`` is a test basis
    function, and ``H_T`` is the modified Hilbert transform on
    ``I = (0, final_time)``.
    """
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


def get_hilbert_matrix_for_derivatives_legendre_lagrange(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                         derivatives_trial: int, derivatives_test: int,
                                                         final_time: float):
    r"""Return a Hilbert matrix with Legendre trial and Lagrange test basis.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Legendre trial basis function and ``psi_r`` is a
    Lagrange test basis function.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        nt, pol_deg_trial, pol_deg_test, derivatives_trial, derivatives_test, final_time
    )
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_test, nt))

    return trans.T @ K


def get_hilbert_matrix_for_derivatives_lagrange_legendre(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                         derivatives_trial: int, derivatives_test: int,
                                                         final_time: float):
    r"""Return a Hilbert matrix with Lagrange trial and Legendre test basis.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Lagrange trial basis function and ``psi_r`` is a
    Legendre test basis function.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        nt, pol_deg_trial, pol_deg_test, derivatives_trial, derivatives_test, final_time
    )
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_trial, nt))

    return K @ trans


def get_hilbert_matrix_for_derivatives_lagrange_lagrange(nt: int, pol_deg_trial: int, pol_deg_test: int,
                                                         derivatives_trial: int, derivatives_test: int,
                                                         final_time: float):
    r"""Return a Hilbert matrix with Lagrange trial/test bases.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Lagrange trial basis function and ``psi_r`` is a
    Lagrange test basis function. This is the common form used, for example,
    in the hybrid ODE bilinear form.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        nt, pol_deg_trial, pol_deg_test, derivatives_trial, derivatives_test, final_time
    )
    trans = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_trial, nt))
    transT = sparse.linalg.aslinearoperator(sm.get_lagrange_to_legendre_matrix(pol_deg_test, nt))

    return transT.T @ K @ trans


# Backwards-compatible aliases for the previous public names.
get_hilbert_matrix_with_derivatives_legendre_legendre = get_hilbert_matrix_for_derivatives_legendre_legendre
get_hilbert_matrix_with_derivatives_legendre_lagrange = get_hilbert_matrix_for_derivatives_legendre_lagrange
get_hilbert_matrix_with_derivatives_lagrange_legendre = get_hilbert_matrix_for_derivatives_lagrange_legendre
get_hilbert_matrix_with_derivatives_lagrange_lagrange = get_hilbert_matrix_for_derivatives_lagrange_lagrange
