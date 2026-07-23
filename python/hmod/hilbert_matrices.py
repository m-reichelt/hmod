import scipy.sparse as sparse
from scipy.sparse.linalg import LinearOperator
import numpy as np
import hmod.block_operations as block_ops
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


class DFT(LinearOperator):
    """Unnormalized forward discrete Fourier transform on ``nt`` values."""

    def __init__(self, nt: int, workers=-1):
        self.nt = nt
        self.workers = workers
        super().__init__(dtype=np.complex128, shape=(nt, nt))

    def _matmat(self, x):
        from scipy.fft import fft
        return fft(x, axis=0, workers=self.workers)

    def _rmatmat(self, x):
        from scipy.fft import ifft
        return self.nt * ifft(x, axis=0, workers=self.workers)


class _RealLinearOperator(LinearOperator):
    """Expose an analytically real operator assembled with complex FFTs."""

    def __init__(self, operator):
        self.operator = sparse.linalg.aslinearoperator(operator)
        super().__init__(dtype=np.float64, shape=self.operator.shape)

    @staticmethod
    def _apply_real_operator(operator, x):
        x = np.asarray(x)
        if np.iscomplexobj(x):
            return np.real(operator @ x.real) + 1j * np.real(operator @ x.imag)
        return np.real(operator @ x)

    def _matmat(self, x):
        return self._apply_real_operator(self.operator, x)

    def _matvec(self, x):
        return self._apply_real_operator(self.operator, x)

    def _rmatmat(self, x):
        return self._apply_real_operator(self.operator.H, x)

    def _rmatvec(self, x):
        return self._apply_real_operator(self.operator.H, x)


def get_trial_transform(polynomial_degree, nt, workers=-1, periodic=False):
    """Return the FFT-based block transform for trial Legendre modes.

    The transform maps degree-major piecewise Legendre coefficients into
    the sine/cosine coefficient ordering used by the modified Hilbert
    transform kernel. Even trial degrees use DCT-IV blocks, odd trial
    degrees use DST-IV blocks, with the signs from the analytic formula.

    For ``periodic=True``, every degree block instead uses the ordinary,
    unnormalized length-``nt`` DFT from the periodic factorization.
    """
    blocks = [[None for _ in range(polynomial_degree + 1)] for _ in range(polynomial_degree + 1)]

    for m in range(polynomial_degree + 1):
        if periodic:
            blocks[m][m] = DFT(nt, workers=workers)
        elif m % 2 == 0:
            fac = np.power(-1, m // 2)
            blocks[m][m] = fac * DCT_IV(nt, workers=workers)
        else:
            fac = np.power(-1, (m + 1) // 2)
            blocks[m][m] = fac * DST_IV(nt, workers=workers)

    block_operator = block_ops.BlockLinearOperator(blocks)
    return block_operator


def get_test_transform(polynomial_degree, nt, workers=-1, periodic=False):
    """Return the FFT-based block transform for test Legendre modes.

    This is the test-side analogue of :func:`get_trial_transform`. It maps
    piecewise Legendre test coefficients into the sine/cosine coefficient
    ordering used for the action of ``H_T`` on the test function in
    ``<u, H_T v>_I``.

    For ``periodic=True``, every degree block uses the ordinary,
    unnormalized length-``nt`` DFT.
    """
    blocks = [[None for _ in range(polynomial_degree + 1)] for _ in range(polynomial_degree + 1)]

    for m in range(polynomial_degree + 1):
        if periodic:
            blocks[m][m] = DFT(nt, workers=workers)
        elif m % 2 == 0:
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


def first_branch_sum(
    diagonal_index: int,
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    periodic: bool = False,
):
    """Compute the first zeta-series branch for one kernel diagonal entry.

    The returned scalar contributes to the diagonal kernel block coupling
    trial Legendre degree ``polynomial_degree_trial`` with test Legendre degree
    ``polynomial_degree_test`` for the bilinear form ``<u, H_T v>_I``.
    Set ``periodic=True`` to use the first periodic aliasing branch.
    """
    if periodic:
        return _periodic_first_branch_sum(
            diagonal_index,
            nt,
            polynomial_degree_trial,
            polynomial_degree_test,
        ) / nt ** 2

    from scipy.special import spherical_jn as jn
    # use shorthand notation for the indices as in paper
    r = polynomial_degree_test
    m = polynomial_degree_trial
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


def second_branch_sum(
    diagonal_index: int,
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    periodic: bool = False,
):
    """Compute the second zeta-series branch for one kernel diagonal entry.

    Together with :func:`first_branch_sum`, this gives the diagonal kernel
    block for a fixed pair of trial/test Legendre degrees in the matrix of
    ``<u, H_T v>_I``.
    Set ``periodic=True`` to use the reflected periodic aliasing branch.
    """
    if periodic:
        return _periodic_second_branch_sum(
            diagonal_index,
            nt,
            polynomial_degree_trial,
            polynomial_degree_test,
        ) / nt ** 2

    # use shorthand notation for the indices as in paper
    r = polynomial_degree_test
    m = polynomial_degree_trial
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


def _periodic_trigonometric_factors(theta, degree: int):
    sigma = np.sin(theta - degree * np.pi / 2)
    kappa = np.cos(theta - degree * np.pi / 2)
    sigma = np.where(np.abs(sigma) < 8 * np.finfo(float).eps, 0.0, sigma)
    kappa = np.where(np.abs(kappa) < 8 * np.finfo(float).eps, 0.0, kappa)
    return sigma, kappa


def _periodic_bessel_product_tail(
    theta_over_pi,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    reflected: bool,
):
    """Evaluate one periodic Bessel-product tail with Hurwitz zeta sums."""
    m = polynomial_degree_trial
    r = polynomial_degree_test
    theta_over_pi = np.asarray(theta_over_pi, dtype=np.float64)
    theta = np.pi * theta_over_pi

    sigma_r, kappa_r = _periodic_trigonometric_factors(theta, r)
    sigma_m, kappa_m = _periodic_trigonometric_factors(theta, m)

    if reflected:
        zeta_parameter = 2.0 - theta_over_pi
        parity = (-1) ** (r + m)
        cross_sign = -1.0
    else:
        zeta_parameter = 1.0 + theta_over_pi
        parity = 1.0
        cross_sign = 1.0

    result = np.zeros_like(theta_over_pi, dtype=np.float64)
    for i in range(0, r // 2 + 1):
        for j in range(0, m // 2 + 1):
            exponent = 2 * (i + j) + 2
            result += (
                parity
                * _a_m_i(r, i)
                * _a_m_i(m, j)
                * sigma_r
                * sigma_m
                * zeta(exponent, zeta_parameter)
                / np.pi ** exponent
            )
    for i in range(0, r // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            exponent = 2 * (i + j) + 3
            result += (
                parity
                * cross_sign
                * _a_m_i(r, i)
                * _b_m_i(m, j)
                * sigma_r
                * kappa_m
                * zeta(exponent, zeta_parameter)
                / np.pi ** exponent
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, m // 2 + 1):
            exponent = 2 * (i + j) + 3
            result += (
                parity
                * cross_sign
                * _b_m_i(r, i)
                * _a_m_i(m, j)
                * kappa_r
                * sigma_m
                * zeta(exponent, zeta_parameter)
                / np.pi ** exponent
            )
    for i in range(0, (r - 1) // 2 + 1):
        for j in range(0, (m - 1) // 2 + 1):
            exponent = 2 * (i + j) + 4
            result += (
                parity
                * _b_m_i(r, i)
                * _b_m_i(m, j)
                * kappa_r
                * kappa_m
                * zeta(exponent, zeta_parameter)
                / np.pi ** exponent
            )
    return result


def _periodic_zero_branch_sum(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
):
    """Return ``sum_{q>=1} j_r(pi*q) j_m(pi*q)`` exactly."""
    return float(
        _periodic_bessel_product_tail(
            0.0,
            polynomial_degree_trial,
            polynomial_degree_test,
            reflected=False,
        )
    )


def _periodic_first_branch_sum(
    diagonal_index: int,
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
):
    if not 0 <= diagonal_index < nt:
        raise ValueError("diagonal_index must satisfy 0 <= diagonal_index < nt")
    if diagonal_index == 0:
        # The continuous zero mode is excluded from the periodic multiplier.
        return _periodic_zero_branch_sum(
            polynomial_degree_trial, polynomial_degree_test
        )

    theta_over_pi = diagonal_index / nt
    theta = np.pi * theta_over_pi
    direct = (
        spherical_jn(polynomial_degree_test, theta)
        * spherical_jn(polynomial_degree_trial, theta)
    )
    tail = _periodic_bessel_product_tail(
        theta_over_pi,
        polynomial_degree_trial,
        polynomial_degree_test,
        reflected=False,
    )
    return float(direct + tail)


def _periodic_second_branch_sum(
    diagonal_index: int,
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
):
    if not 0 <= diagonal_index < nt:
        raise ValueError("diagonal_index must satisfy 0 <= diagonal_index < nt")
    if diagonal_index == 0:
        return _periodic_zero_branch_sum(
            polynomial_degree_trial, polynomial_degree_test
        )

    theta_over_pi = diagonal_index / nt
    reflected_argument = np.pi * (1.0 - theta_over_pi)
    direct = (
        spherical_jn(polynomial_degree_test, reflected_argument)
        * spherical_jn(polynomial_degree_trial, reflected_argument)
    )
    tail = _periodic_bessel_product_tail(
        theta_over_pi,
        polynomial_degree_trial,
        polynomial_degree_test,
        reflected=True,
    )
    return float(direct + tail)


def periodic_aliasing_sum(
    diagonal_index: int,
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
):
    r"""Return the periodic aliasing sum ``Gamma_a^{r,m}`` from the paper."""
    if nt <= 0:
        raise ValueError("nt must be positive")
    if not 0 <= diagonal_index < nt:
        raise ValueError("diagonal_index must satisfy 0 <= diagonal_index < nt")

    if diagonal_index == 0:
        parity = (-1) ** (
            polynomial_degree_trial + polynomial_degree_test
        )
        return (1.0 - parity) * _periodic_zero_branch_sum(
            polynomial_degree_trial, polynomial_degree_test
        )

    parity = (-1) ** (
        polynomial_degree_trial + polynomial_degree_test
    )
    return _periodic_first_branch_sum(
        diagonal_index,
        nt,
        polynomial_degree_trial,
        polynomial_degree_test,
    ) - parity * _periodic_second_branch_sum(
        diagonal_index,
        nt,
        polynomial_degree_trial,
        polynomial_degree_test,
    )


def _periodic_kernel_diagonal_entries_zeta(
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
):
    """Vectorized periodic symbols ``Lambda_a^{r,m}`` for one degree pair."""
    if nt <= 0:
        raise ValueError("nt must be positive")

    m = polynomial_degree_trial
    r = polynomial_degree_test
    gamma = np.zeros(nt, dtype=np.float64)
    parity = (-1) ** (r + m)

    gamma[0] = (1.0 - parity) * _periodic_zero_branch_sum(m, r)
    if nt > 1:
        theta_over_pi = np.arange(1, nt, dtype=np.float64) / nt
        theta = np.pi * theta_over_pi
        first = (
            spherical_jn(r, theta) * spherical_jn(m, theta)
            + _periodic_bessel_product_tail(
                theta_over_pi, m, r, reflected=False
            )
        )
        reflected_argument = np.pi * (1.0 - theta_over_pi)
        second = (
            spherical_jn(r, reflected_argument)
            * spherical_jn(m, reflected_argument)
            + _periodic_bessel_product_tail(
                theta_over_pi, m, r, reflected=True
            )
        )
        gamma[1:] = first - parity * second

    return -(1j ** (r - m + 1)) * gamma / nt ** 2


def _kernel_diagonal_entries_zeta(
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    periodic: bool = False,
):
    """Vectorized diagonal entries for one transformed Hilbert kernel block."""
    if periodic:
        return _periodic_kernel_diagonal_entries_zeta(
            nt, polynomial_degree_trial, polynomial_degree_test
        )

    r = polynomial_degree_test
    m = polynomial_degree_trial

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
    if polynomial_degree_trial % 2 != polynomial_degree_test % 2:
        second_branch_sign = 1.0

    return (first + second_branch_sign * second) / nt ** 2


def get_kernel_matrix_for_degrees_zeta(
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    periodic: bool = False,
):
    """Return one diagonal kernel block for fixed Legendre degrees.

    The block has shape ``(nt, nt)`` and couples all time intervals for one
    trial degree and one test degree. It is diagonal in the transformed
    sine/cosine basis; the entries are evaluated with the zeta-series
    formula used for the modified Hilbert transform. With ``periodic=True``,
    the entries are the complex ordinary-FFT symbols of the periodic
    transform.
    """
    entries = _kernel_diagonal_entries_zeta(
        nt,
        polynomial_degree_trial,
        polynomial_degree_test,
        periodic=periodic,
    )
    return sparse.diags(entries, format='csr')


def get_kernel_matrix(
    nt: int,
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    periodic: bool = False,
):
    """Return the full transformed kernel for all Legendre degree pairs.

    The result is a block matrix collecting
    :func:`get_kernel_matrix_for_degrees_zeta` for trial degrees
    ``0..polynomial_degree_trial`` and test degrees ``0..polynomial_degree_test``.
    """
    blocks = [[get_kernel_matrix_for_degrees_zeta(nt, n, m, periodic=periodic) for n in range(polynomial_degree_trial + 1)]
              for m in
              range(polynomial_degree_test + 1)]
    return sparse.bmat(blocks, format='csr')


def get_operator_I_H_legendre_legendre(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    r"""Return the Legendre-Legendre operator for ``<u, H_T v>_I``.

    Rows correspond to test functions and columns to trial functions. For
    trial basis functions ``phi_m`` and test basis functions ``psi_r``,
    this operator represents

    ``A[r, m] = <phi_m, H_T psi_r>_I``.

    The returned object is matrix-free and uses FFT-based transforms around
    the zeta-series kernel. ``periodic=False`` selects the modified transform
    with type-IV sine/cosine transforms; ``periodic=True`` selects
    ``H_per`` with ordinary complex FFTs.
    """
    trial_transform = get_trial_transform(
        polynomial_degree_trial, nt, periodic=periodic
    )
    test_transform = get_test_transform(
        polynomial_degree_test, nt, periodic=periodic
    )
    K = sparse.linalg.aslinearoperator(
        get_kernel_matrix(
            nt,
            polynomial_degree_trial,
            polynomial_degree_test,
            periodic=periodic,
        )
    )
    operator = test_transform.H @ K @ trial_transform
    if periodic:
        return _RealLinearOperator(T * operator)
    return (T * 0.5) * operator


def get_hilbert_matrix_for_derivatives_legendre_legendre(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    derivatives_trial: int,
    derivatives_test: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    r"""Return a Hilbert matrix in Legendre trial/test bases.

    Rows correspond to test basis functions and columns to trial basis
    functions. With ``i = derivatives_trial`` and
    ``j = derivatives_test``, the matrix entries are

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``.

    Here ``phi_m`` is a trial basis function, ``psi_r`` is a test basis
    function, and ``H_T`` is the modified Hilbert transform on ``I = (0, T)``.
    Set ``periodic=True`` to use the periodic modified Hilbert transform.
    """
    Mh = get_operator_I_H_legendre_legendre(
        polynomial_degree_trial,
        polynomial_degree_test,
        nt,
        T,
        periodic=periodic,
    )
    Mat = Mh
    if derivatives_trial > 0:
        D_trial = pb.get_legendre_derivative_matrix(
            nt,
            polynomial_degree_trial,
            T,
            periodic=periodic,
        )
        D_pow = sparse.linalg.matrix_power(D_trial, derivatives_trial)
        D_op = sparse.linalg.aslinearoperator(D_pow)
        Mat = Mat @ D_op
    if derivatives_test > 0:
        D_test = pb.get_legendre_derivative_matrix(
            nt,
            polynomial_degree_test,
            T,
            periodic=periodic,
        )
        D_pow = sparse.linalg.matrix_power(D_test, derivatives_test)
        D_op = sparse.linalg.aslinearoperator(D_pow)
        Mat = D_op.H @ Mat

    return Mat


def get_hilbert_matrix_for_derivatives_legendre_lagrange(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    derivatives_trial: int,
    derivatives_test: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    r"""Return a Hilbert matrix with Legendre trial and Lagrange test basis.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Legendre trial basis function and ``psi_r`` is a
    Lagrange test basis function.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        polynomial_degree_trial,
        polynomial_degree_test,
        derivatives_trial,
        derivatives_test,
        nt,
        T,
        periodic=periodic,
    )
    trans = sparse.linalg.aslinearoperator(
        pb.get_lagrange_to_legendre_matrix(
            polynomial_degree_test, nt, periodic=periodic
        )
    )

    return trans.T @ K


def get_hilbert_matrix_for_derivatives_lagrange_legendre(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    derivatives_trial: int,
    derivatives_test: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    r"""Return a Hilbert matrix with Lagrange trial and Legendre test basis.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Lagrange trial basis function and ``psi_r`` is a
    Legendre test basis function.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        polynomial_degree_trial,
        polynomial_degree_test,
        derivatives_trial,
        derivatives_test,
        nt,
        T,
        periodic=periodic,
    )
    trans = sparse.linalg.aslinearoperator(
        pb.get_lagrange_to_legendre_matrix(
            polynomial_degree_trial, nt, periodic=periodic
        )
    )

    return K @ trans


def get_hilbert_matrix_for_derivatives_lagrange_lagrange(
    polynomial_degree_trial: int,
    polynomial_degree_test: int,
    derivatives_trial: int,
    derivatives_test: int,
    nt: int,
    T: float,
    periodic: bool = False,
):
    r"""Return a Hilbert matrix with Lagrange trial/test bases.

    With ``i = derivatives_trial`` and ``j = derivatives_test``, the matrix
    represents

    ``A[r, m] = <partial_t^i phi_m, H_T partial_t^j psi_r>_I``,

    where ``phi_m`` is a Lagrange trial basis function and ``psi_r`` is a
    Lagrange test basis function. This is the common form used, for example,
    in the hybrid ODE bilinear form.
    """
    K = get_hilbert_matrix_for_derivatives_legendre_legendre(
        polynomial_degree_trial,
        polynomial_degree_test,
        derivatives_trial,
        derivatives_test,
        nt,
        T,
        periodic=periodic,
    )
    trans = sparse.linalg.aslinearoperator(
        pb.get_lagrange_to_legendre_matrix(
            polynomial_degree_trial, nt, periodic=periodic
        )
    )
    transT = sparse.linalg.aslinearoperator(
        pb.get_lagrange_to_legendre_matrix(
            polynomial_degree_test, nt, periodic=periodic
        )
    )

    return transT.T @ K @ trans


# Backwards-compatible aliases for the previous function names.
get_hilbert_matrix_with_derivatives_legendre_legendre = get_hilbert_matrix_for_derivatives_legendre_legendre
get_hilbert_matrix_with_derivatives_legendre_lagrange = get_hilbert_matrix_for_derivatives_legendre_lagrange
get_hilbert_matrix_with_derivatives_lagrange_legendre = get_hilbert_matrix_for_derivatives_lagrange_legendre
get_hilbert_matrix_with_derivatives_lagrange_lagrange = get_hilbert_matrix_for_derivatives_lagrange_lagrange
