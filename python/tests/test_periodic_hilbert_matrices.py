"""Independent checks for the periodic modified-Hilbert implementation.

The reference calculations in this module deliberately use direct spherical
Bessel and Fourier sums instead of the Hurwitz-zeta kernel factorization under
test.  The sums are vectorized so that the fairly large truncation parameters
still keep the tests inexpensive.
"""

import numpy as np
import pytest
from scipy.special import spherical_jn

import hmod.hilbert_matrices as hm


_BRANCH_TERMS = 200_000
_ZERO_BIN_TERMS = 20_000
_DENSE_FOURIER_CUTOFF = 20_000


def _materialize(operator):
    """Materialize a possibly matrix-free operator in column order."""
    return np.asarray(operator @ np.eye(operator.shape[1]))


def _direct_periodic_branches(nt, diagonal_index, trial_degree, test_degree):
    """Truncated versions of the two positive-argument sums in the paper."""
    assert 0 < diagonal_index < nt
    q = np.arange(_BRANCH_TERMS, dtype=np.float64)
    theta = np.pi * diagonal_index / nt

    plus_arguments = np.pi * q + theta
    minus_arguments = np.pi * (q + 1.0) - theta
    plus = np.sum(
        spherical_jn(test_degree, plus_arguments)
        * spherical_jn(trial_degree, plus_arguments)
    )
    minus = np.sum(
        spherical_jn(test_degree, minus_arguments)
        * spherical_jn(trial_degree, minus_arguments)
    )
    return plus, minus


def _direct_zero_bin_aliasing_sum(trial_degree, test_degree):
    """Directly pair the positive and negative nonzero modes in FFT bin zero."""
    q = np.arange(1, _ZERO_BIN_TERMS + 1, dtype=np.float64)
    arguments = np.pi * q
    parity = (-1) ** (trial_degree + test_degree)
    return (1.0 - parity) * np.sum(
        spherical_jn(test_degree, arguments)
        * spherical_jn(trial_degree, arguments)
    )


def _spherical_jn_on_signed_arguments(degree, arguments):
    """Evaluate j_degree(x), including its parity for negative real x."""
    values = spherical_jn(degree, np.abs(arguments))
    if degree % 2:
        values = np.where(arguments < 0.0, -values, values)
    return values


def _direct_truncated_fourier_matrix(nt, polynomial_degree, T):
    """Dense Legendre matrix from the unfactored bilateral Fourier sum."""
    cutoff = _DENSE_FOURIER_CUTOFF
    frequencies = np.concatenate(
        (
            np.arange(-cutoff, 0, dtype=np.int64),
            np.arange(1, cutoff + 1, dtype=np.int64),
        )
    )
    arguments = np.pi * frequencies / nt
    bessel_values = np.vstack(
        [
            _spherical_jn_on_signed_arguments(degree, arguments)
            for degree in range(polynomial_degree + 1)
        ]
    )

    element_indices = np.arange(nt)
    element_differences = (
        element_indices[:, None] - element_indices[None, :]
    )
    phases = np.exp(
        2j
        * np.pi
        * element_differences[..., None]
        * frequencies
        / nt
    )

    size = nt * (polynomial_degree + 1)
    result = np.zeros((size, size), dtype=np.complex128)
    frequency_signs = np.sign(frequencies)
    for test_degree in range(polynomial_degree + 1):
        row_slice = slice(test_degree * nt, (test_degree + 1) * nt)
        for trial_degree in range(polynomial_degree + 1):
            col_slice = slice(trial_degree * nt, (trial_degree + 1) * nt)
            weights = (
                frequency_signs
                * bessel_values[test_degree]
                * bessel_values[trial_degree]
            )
            result[row_slice, col_slice] = (
                -T
                * (1j ** (test_degree - trial_degree + 1))
                / nt**2
                * np.sum(phases * weights, axis=-1)
            )
    return result


@pytest.mark.parametrize(
    "nt, diagonal_index, trial_degree, test_degree",
    [
        (5, 1, 0, 0),
        (5, 4, 2, 3),
        (7, 3, 4, 1),
    ],
)
def test_periodic_branches_and_aliasing_match_direct_bessel_sums(
    nt, diagonal_index, trial_degree, test_degree
):
    direct_plus, direct_minus = _direct_periodic_branches(
        nt, diagonal_index, trial_degree, test_degree
    )

    # The public branch helpers retain the legacy 1/nt**2 kernel scaling.
    computed_plus = hm.first_branch_sum(
        diagonal_index,
        nt,
        trial_degree,
        test_degree,
        periodic=True,
    )
    computed_minus = hm.second_branch_sum(
        diagonal_index,
        nt,
        trial_degree,
        test_degree,
        periodic=True,
    )
    np.testing.assert_allclose(
        computed_plus, direct_plus / nt**2, rtol=0.0, atol=1.1e-8
    )
    np.testing.assert_allclose(
        computed_minus, direct_minus / nt**2, rtol=0.0, atol=1.1e-8
    )

    parity = (-1) ** (trial_degree + test_degree)
    direct_gamma = direct_plus - parity * direct_minus
    computed_gamma = hm.periodic_aliasing_sum(
        diagonal_index, nt, trial_degree, test_degree
    )
    np.testing.assert_allclose(
        computed_gamma, direct_gamma, rtol=0.0, atol=1.0e-9
    )


def test_zero_fft_bin_has_the_nonzero_mixed_degree_alias():
    nt = 7
    trial_degree = 1
    test_degree = 2

    direct_gamma = _direct_zero_bin_aliasing_sum(
        trial_degree, test_degree
    )
    computed_gamma = hm.periodic_aliasing_sum(
        0, nt, trial_degree, test_degree
    )
    np.testing.assert_allclose(
        computed_gamma, direct_gamma, rtol=0.0, atol=3.0e-10
    )
    assert abs(computed_gamma) > 0.1

    symbol = hm.get_kernel_matrix_for_degrees_zeta(
        nt,
        trial_degree,
        test_degree,
        periodic=True,
    ).diagonal()
    expected_zero_symbol = (
        -(1j ** (test_degree - trial_degree + 1))
        * computed_gamma
        / nt**2
    )
    np.testing.assert_allclose(
        symbol[0], expected_zero_symbol, rtol=0.0, atol=1.0e-14
    )
    assert abs(symbol[0]) > 1.0e-3
    assert abs(symbol[0].imag) < 1.0e-14


@pytest.mark.parametrize(
    "trial_degree, test_degree",
    [(0, 0), (1, 2), (1, 3)],
)
def test_periodic_alias_and_symbol_symmetries(
    trial_degree, test_degree
):
    nt = 8
    diagonal_index = 3
    reflected_index = nt - diagonal_index
    parity = (-1) ** (trial_degree + test_degree)

    gamma = hm.periodic_aliasing_sum(
        diagonal_index, nt, trial_degree, test_degree
    )
    gamma_degree_transpose = hm.periodic_aliasing_sum(
        diagonal_index, nt, test_degree, trial_degree
    )
    gamma_reflected = hm.periodic_aliasing_sum(
        reflected_index, nt, trial_degree, test_degree
    )
    np.testing.assert_allclose(
        gamma_degree_transpose, gamma, rtol=0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        gamma_reflected,
        (-1) ** (trial_degree + test_degree + 1) * gamma,
        rtol=0.0,
        atol=2.0e-13,
    )

    symbol = hm.get_kernel_matrix_for_degrees_zeta(
        nt, trial_degree, test_degree, periodic=True
    ).diagonal()
    transposed_symbol = hm.get_kernel_matrix_for_degrees_zeta(
        nt, test_degree, trial_degree, periodic=True
    ).diagonal()
    np.testing.assert_allclose(
        symbol[reflected_index],
        np.conj(symbol[diagonal_index]),
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        symbol,
        -np.conj(transposed_symbol),
        rtol=0.0,
        atol=2.0e-13,
    )

    # At the even-grid Nyquist mode, opposite-parity degree couplings are
    # real; same-parity couplings vanish by the reflection identity.
    nyquist_value = symbol[nt // 2]
    if parity == -1:
        assert abs(nyquist_value.imag) < 2.0e-13
    else:
        assert abs(nyquist_value) < 2.0e-13


def test_periodic_legendre_operator_matches_direct_fourier_matrix():
    nt = 3
    polynomial_degree = 2
    T = 1.7

    operator = hm.get_operator_I_H_legendre_legendre(
        polynomial_degree,
        polynomial_degree,
        nt,
        T,
        periodic=True,
    )
    computed = _materialize(operator)
    reference = _direct_truncated_fourier_matrix(
        nt, polynomial_degree, T
    )

    assert np.max(np.abs(reference.imag)) < 1.0e-12
    np.testing.assert_allclose(
        computed, reference.real, rtol=0.0, atol=2.0e-8
    )


def test_periodic_physical_matrix_is_skew_and_annihilates_constants():
    nt = 5
    polynomial_degree = 3
    operator = hm.get_operator_I_H_legendre_legendre(
        polynomial_degree,
        polynomial_degree,
        nt,
        T=2.3,
        periodic=True,
    )
    matrix = _materialize(operator)
    adjoint_matrix = _materialize(operator.H)

    np.testing.assert_allclose(
        matrix + matrix.T,
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint_matrix, matrix.T, rtol=0.0, atol=2.0e-12
    )

    constant = np.zeros(nt * (polynomial_degree + 1))
    constant[:nt] = 1.0
    np.testing.assert_allclose(
        matrix @ constant, 0.0, rtol=0.0, atol=2.0e-12
    )


def test_periodic_derivative_hilbert_form_is_positive_semidefinite():
    nt = 5
    polynomial_degree = 2
    n_dofs = nt * polynomial_degree
    operator = hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(
        polynomial_degree,
        polynomial_degree,
        derivatives_trial=1,
        derivatives_test=0,
        nt=nt,
        T=1.3,
        periodic=True,
    )
    matrix = _materialize(operator)

    np.testing.assert_allclose(
        matrix, matrix.T, rtol=0.0, atol=2.0e-12
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    assert eigenvalues[0] > -2.0e-12
    assert eigenvalues[1] > 1.0e-6

    constant = np.ones(n_dofs)
    np.testing.assert_allclose(
        matrix @ constant, 0.0, rtol=0.0, atol=2.0e-12
    )


@pytest.mark.parametrize(
    "builder, expected_shape",
    [
        (
            hm.get_hilbert_matrix_for_derivatives_legendre_legendre,
            (9, 6),
        ),
        (
            hm.get_hilbert_matrix_for_derivatives_legendre_lagrange,
            (6, 6),
        ),
        (
            hm.get_hilbert_matrix_for_derivatives_lagrange_legendre,
            (9, 3),
        ),
        (
            hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange,
            (6, 3),
        ),
    ],
)
def test_periodic_flag_propagates_through_every_basis_pair(
    builder, expected_shape
):
    operator = builder(
        polynomial_degree_trial=1,
        polynomial_degree_test=2,
        derivatives_trial=0,
        derivatives_test=0,
        nt=3,
        T=1.0,
        periodic=True,
    )

    assert operator.shape == expected_shape


def test_explicit_periodic_false_preserves_nonperiodic_results():
    branch_args = (2, 3, 2, 1)
    assert hm.first_branch_sum(*branch_args) == hm.first_branch_sum(
        *branch_args, periodic=False
    )
    assert hm.second_branch_sum(*branch_args) == hm.second_branch_sum(
        *branch_args, periodic=False
    )

    operator_args = (1, 1, 1, 0, 3, 1.2)
    default_operator = (
        hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(
            *operator_args
        )
    )
    explicit_nonperiodic_operator = (
        hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(
            *operator_args, periodic=False
        )
    )
    np.testing.assert_array_equal(
        _materialize(default_operator),
        _materialize(explicit_nonperiodic_operator),
    )
