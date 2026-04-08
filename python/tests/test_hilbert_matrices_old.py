import pytest
from hmod.deprecated_hilbert_matrices import Rhs_I_H_Lagrange_Lagrange, Operator_dt_H_Lagrange_Lagrange, Operator_I_H_Lagrange_Lagrange
import numpy as np
import scipy.sparse as sp
from ngsolve import *
from ngsolve.meshes import Make1DMesh

def ngsolve_mass_matrix(tpoints):

    nt = len(tpoints) - 1
    mesh = Make1DMesh(n=nt)
    V = H1(mesh, order=1)  # Linear elements with zero at t=0

    a = BilinearForm(V)
    a += InnerProduct(V.TrialFunction(), V.TestFunction()) * dx
    a.Assemble()

    rows,cols,vals = a.mat.COO()
    import scipy.sparse as sp
    M = sp.csr_matrix((vals,(rows,cols)))

    return M

def test_rhs_against_dense():
    import hilbertWrapper.hilbertWrapper as hw
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
    #build the dense rhs matrix using hilbert wrapper
    Fh = hw.get_rhs_mat(tpoints)
    rhs_dense = Fh @ f_vec
    diff_vec = rhs_dense - rhs_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4


def test_dt_H_against_dense():
    import hilbertWrapper.hilbertWrapper as hw
    from hmod.matrix_tools import linear_operator_to_matrix
    f_analytic = lambda t: np.cos(2 * np.pi * t+5)
    nt = 5
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    At_fft = Operator_dt_H_Lagrange_Lagrange(nmodes=40 * nt, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    At_fft_dense = linear_operator_to_matrix(At_fft)
    M = np.array(ngsolve_mass_matrix(tpoints).todense())
    #build the dense dt_H matrix using hilbert wrapper
    Ath = hw.get_hilbert_stiffness_heat(tpoints)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    x[0] = 0.0  #enforce zero at t=0 as first column can be rubbish (either in old or new version)
    dtH_fft = At_fft @ x
    dtH_dense = Ath @ x
    diff_vec = dtH_dense - dtH_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4

def test_I_H_against_dense():
    import hilbertWrapper.hilbertWrapper as hw
    f_analytic = lambda t: np.cos(2 * np.pi * t+5)
    nt = 5
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    n_modes = 8*nt
    Mt_fft = Operator_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #do it also via rhs
    rhs_builder = Rhs_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_rhs=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #build the dense dt_H matrix using hilbert wrapper
    Mth = hw.get_hilbert_mass(tpoints)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = Mth @ x
    I_H_rhs = rhs_builder @ (Mt_fft.Ttrial @ x)
    diff_vec = I_H_dense - I_H_fft
    diff = np.linalg.norm(diff_vec)
    assert diff < 1e-4

def test_mat_mat():
    import hilbertWrapper.hilbertWrapper as hw
    f_analytic = lambda t: np.cos(2 * np.pi * t+5)
    nt = 5
    T = 1.0
    tpoints = np.linspace(0, T, nt+1)
    polynomial_degree_trial = 1
    polynomial_degree_test = 1
    n_modes = 8*nt
    Mt_fft = Operator_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_trial=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #do it also via rhs
    rhs_builder = Rhs_I_H_Lagrange_Lagrange(nmodes=n_modes, nt=nt, polynomial_degree_rhs=polynomial_degree_trial, polynomial_degree_test=polynomial_degree_test)
    #build the dense dt_H matrix using hilbert wrapper
    Mth = hw.get_hilbert_mass(tpoints)
    #get a random vector
    np.random.seed(0)
    x = np.random.rand(nt*(polynomial_degree_trial)+1, 5).astype(np.float64)
    I_H_fft = Mt_fft @ x
    I_H_dense = Mth @ x
    I_H_rhs = rhs_builder @ (Mt_fft.Ttrial @ x)


if __name__ == "__main__":
    pytest.main([__file__])
