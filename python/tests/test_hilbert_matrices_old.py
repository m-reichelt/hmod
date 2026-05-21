import pytest
from hmod.deprecated_hilbert_matrices import Rhs_I_H_Lagrange_Lagrange, Operator_dt_H_Lagrange_Lagrange, Operator_I_H_Lagrange_Lagrange
import numpy as np
from hilbert_wrapper_reference_data import (
    DENSE_HILBERT_MASS_NT5,
    DENSE_HILBERT_STIFFNESS_HEAT_NT5,
    DENSE_RHS_COS_PHASE5_NT50,
)

def test_rhs_against_dense():
    f_analytic = lambda t: np.cos(2 * np.pi * t+5)
    nt = 50
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    f_vec = 0.5 * (f_analytic(tpoints[:-1]) + f_analytic(tpoints[1:]))
    polynomial_degree_test = 1
    polynomial_degree_rhs = 0
    n_modes = 100*nt
    rhs_builder = Rhs_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_rhs=polynomial_degree_rhs, polynomial_degree_test=polynomial_degree_test)
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
    At_fft = Operator_dt_H_Lagrange_Lagrange(nmodes=40 * nt, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    x[0] = 0.0  #enforce zero at t=0 as first column can be rubbish (either in old or new version)
    dtH_fft = At_fft @ x
    dtH_dense = DENSE_HILBERT_STIFFNESS_HEAT_NT5 @ x
    diff_vec = dtH_dense - dtH_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4

def test_I_H_against_dense():
    nt = 5
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    n_modes = 8*nt
    Mt_fft = Operator_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #do it also via rhs
    rhs_builder = Rhs_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_rhs=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = DENSE_HILBERT_MASS_NT5 @ x
    I_H_rhs = rhs_builder @ (Mt_fft.Ttrial @ x)
    diff_vec = I_H_dense - I_H_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4

def test_mat_mat():
    nt = 5
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    n_modes = 8*nt
    Mt_fft = Operator_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #do it also via rhs
    rhs_builder = Rhs_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_rhs=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1, 5).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = DENSE_HILBERT_MASS_NT5 @ x
    I_H_rhs = rhs_builder @ (Mt_fft.Ttrial @ x)


if __name__ == "__main__":
    pytest.main([__file__])
