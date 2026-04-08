import pytest
import numpy as np
import hmod.hilbert_matrices as hm
import matplotlib.pyplot as plt
import hmod.deprecated_hilbert_matrices as dhm

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

def test_kernel_matrix():
    nt = 50
    pol_deg_trial= 2
    pol_deg_test = 2
    K = hm.get_kernel_matrix(nt, pol_deg_trial, pol_deg_test)
    Kd = K.toarray()
    nmodes = int(2e5)
    fourier_facs = [1. for i in range(nmodes)]
    Op = dhm.Base_Operator_Cosine_Sine_Legendre(fourier_facs, nmodes, nt, pol_deg_trial, pol_deg_test)
    K_op = Op.K
    K_opD = (2./nt**2)*K_op.toarray() #we need a factor of 2/nt**2 to match old implementation
    Diff = Kd - K_opD
    assert np.allclose(Kd, K_opD, atol=1e-6, rtol=1e-5)
    assert K.shape == (nt*(pol_deg_test+1), nt*(pol_deg_trial+1))


def test_I_H_against_dense():
    import hilbertWrapper.hilbertWrapper as hw
    nt = 1
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    Mt_fft = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(nt, polynomial_degree_trial, polynomial_degree_test, 0, 0, T)
    #build the dense dt_H matrix using hilbert wrapper
    Mth = hw.get_hilbert_mass(tpoints)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = Mth @ x
    diff_vec = I_H_dense - I_H_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4