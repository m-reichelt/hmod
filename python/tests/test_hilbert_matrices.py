import pytest
import numpy as np
import hmod.hilbert_matrices as hm
import matplotlib.pyplot as plt

from hilbert_wrapper_reference_data import (
    DENSE_HILBERT_MASS_NT1,
    DENSE_HILBERT_STIFFNESS_HEAT_NT5,
    DENSE_RHS_COS_PHASE5_NT50,
)


def _first_branch_sum_brute_force(diagonal_index: int, nt: int, pol_deg_trial: int, pol_deg_test: int):
    assert 0 <= diagonal_index < nt, "Diagonal index must be between 0 and nt-1"
    from scipy.special import spherical_jn as jn

    alpha_k = lambda k: np.pi * (2 * k + 1) / (4 * nt)
    alpha_q = lambda q: alpha_k(2 * q * nt + diagonal_index)
    q_vals = np.arange(int(1e5))[::-1]
    summand = jn(pol_deg_test, alpha_q(q_vals)) * jn(pol_deg_trial, alpha_q(q_vals))
    return np.sum(summand) / (nt ** 2)


def _second_branch_sum_brute_force(diagonal_index: int, nt: int, pol_deg_trial: int, pol_deg_test: int):
    assert 0 <= diagonal_index < nt, "Diagonal index must be between 0 and nt-1"
    from scipy.special import spherical_jn as jn

    alpha_k = lambda k: np.pi * (2 * k + 1) / (4 * nt)
    alpha_q = lambda q: alpha_k(2 * nt * (q + 1) - 1 - diagonal_index)
    q_vals = np.arange(int(1e5))[::-1]
    summand = jn(pol_deg_test, alpha_q(q_vals)) * jn(pol_deg_trial, alpha_q(q_vals))
    return np.sum(summand) / (nt ** 2)


def test_trial_transform():
    nt = 5
    pol_deg = 3
    workers = -1
    U = hm.get_trial_transform(pol_deg, nt, workers=workers)

    #get vector and try multiplication
    x = np.random.rand(nt*(pol_deg+1))
    Ux = U @ x
    Uhx = U.H @ x
    #get matrix and try multiplication
    X = np.random.rand(nt*(pol_deg+1), 3)
    UX = U @ X
    UHX = U.H @ X

    assert U.shape == (nt*(pol_deg+1), nt*(pol_deg+1))

def test_test_transform():
    nt = 5
    pol_deg = 3
    workers = -1
    T = hm.get_test_transform(pol_deg, nt, workers=workers)

    #get vector and try multiplication
    x = np.random.rand(nt*(pol_deg+1))
    Tx = T @ x
    Thx = T.H @ x
    #get matrix and try multiplication
    X = np.random.rand(nt*(pol_deg+1), 3)
    TX = T @ X
    THX = T.H @ X

    assert T.shape == (nt*(pol_deg+1), nt*(pol_deg+1))


def test_first_branch_sum():
    cases = [
        (5, 4, 3, 2),
        (5, 5, 5, 0),
    ]
    for pol_deg_trial, pol_deg_test, nt, diagonal_index in cases:
        first_sum_brute_force = _first_branch_sum_brute_force(diagonal_index, nt, pol_deg_trial, pol_deg_test)
        first_sum = hm.first_branch_sum(diagonal_index, nt, pol_deg_trial, pol_deg_test)
        assert np.allclose(first_sum_brute_force, first_sum, atol=1e-5)

def test_second_branch_sum():
    pol_deg_trial = 5
    pol_deg_test = 3
    nt = 3
    diagonal_index = 2
    first_sum_brute_force = _second_branch_sum_brute_force(diagonal_index, nt, pol_deg_trial, pol_deg_test)
    first_sum = hm.second_branch_sum(diagonal_index, nt, pol_deg_trial, pol_deg_test)
    assert np.allclose(first_sum_brute_force, first_sum, atol=1e-5)


def test_kernel_matrix_for_degrees_zeta_matches_scalar_branch_sums():
    cases = [
        (7, 5, 3),
        (8, 4, 4),
        (9, 2, 5),
    ]
    for nt, pol_deg_trial, pol_deg_test in cases:
        second_branch_sign = -1.0
        if pol_deg_trial % 2 != pol_deg_test % 2:
            second_branch_sign = 1.0

        expected = np.array([
            hm.first_branch_sum(i, nt, pol_deg_trial, pol_deg_test)
            + second_branch_sign * hm.second_branch_sum(i, nt, pol_deg_trial, pol_deg_test)
            for i in range(nt)
        ])
        actual = hm.get_kernel_matrix_for_degrees_zeta(
            nt, pol_deg_trial, pol_deg_test
        ).diagonal()

        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_I_H_against_dense():
    nt = 1
    T = 1.0
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    Mt_fft = hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(nt, polynomial_degree_trial, polynomial_degree_test, 0, 0, T)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = DENSE_HILBERT_MASS_NT1 @ x
    diff_vec = I_H_dense - I_H_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4

def test_rhs_against_dense():
    f_analytic = lambda t: np.cos(2 * np.pi * t+5)
    nt = 50
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    f_vec = 0.5 * (f_analytic(tpoints[:-1]) + f_analytic(tpoints[1:]))
    polynomial_degree_test = 1
    polynomial_degree_rhs = 0
    rhs_builder = hm.get_hilbert_matrix_for_derivatives_legendre_lagrange(nt, polynomial_degree_rhs, polynomial_degree_test, 0, 0, T)
    rhs_fft = rhs_builder @ f_vec
    rhs_dense = DENSE_RHS_COS_PHASE5_NT50
    diff_vec = rhs_dense - rhs_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4


def test_dt_H_against_dense():
    nt = 5
    T = 1.0
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    At_fft = hm.get_hilbert_matrix_for_derivatives_lagrange_lagrange(nt, polynomial_degree_trial, polynomial_degree_test, 1, 0, T)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    x[0] = 0.0  #enforce zero at t=0 as first column can be rubbish (either in old or new version)
    dtH_fft = At_fft @ x
    dtH_dense = DENSE_HILBERT_STIFFNESS_HEAT_NT5 @ x
    diff_vec = dtH_dense - dtH_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4


if __name__ == "__main__":
    pytest.main([__file__])
