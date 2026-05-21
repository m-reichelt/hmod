import pytest
import numpy as np
import hmod.hilbert_matrices as hm
import matplotlib.pyplot as plt
import hmod.deprecated_hilbert_matrices as dhm
import hmod.matrix_tools as mt


def _test_I_H(pol_deg_trial, pol_deg_test, nt):
    T = 1.0
    nmodes = int(2e5)
    Mt_new = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(nt, pol_deg_trial, pol_deg_test, 0, 0, T)
    Mt_old = dhm.Operator_I_H_Lagrange_Lagrange(nmodes=nmodes, nt=nt, polynomial_degree_trial=pol_deg_trial, polynomial_degree_test=pol_deg_test)
    #trasform to dense operators for testing
    Mt_new_dense = mt.linear_operator_to_matrix(Mt_new)
    Mt_old_dense = mt.linear_operator_to_matrix(Mt_old)
    #compare the two
    Diff = Mt_new_dense - Mt_old_dense
    #assert np.allclose(Mt_new_dense, Mt_old_dense, atol=1e-6)
    if not np.allclose(Mt_new_dense, Mt_old_dense, atol=1e-5):
        print(f"Difference between new and old implementation for pol_deg_trial={pol_deg_trial}, pol_deg_test={pol_deg_test}, nt={nt}:")
        print(Diff)
        
def _test_dt_H(pol_deg_trial, pol_deg_test, nt):
    T = 1.0
    nmodes = int(2e5)
    At_new = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(nt, pol_deg_trial, pol_deg_test, 1, 0, T)
    At_old = dhm.Operator_dt_H_Lagrange_Lagrange(nmodes=nmodes, nt=nt, polynomial_degree_trial=pol_deg_trial, polynomial_degree_test=pol_deg_test)
    #trasform to dense operators for testing
    At_new_dense = mt.linear_operator_to_matrix(At_new)[1:, 1:] #we need to remove the first row and columns as old implementation does only work for homogeneous initial conditions
    At_old_dense = mt.linear_operator_to_matrix(At_old)[1:, 1:] #we need to remove the first row and columns as old implementation does only work for homogeneous initial conditions
    #compare the two
    Diff = At_new_dense - At_old_dense
    #assert np.allclose(Mt_new_dense, Mt_old_dense, atol=1e-6)
    if not np.allclose(At_new_dense, At_old_dense, atol=1e-5):
        print(f"Difference between new and old implementation for pol_deg_trial={pol_deg_trial}, pol_deg_test={pol_deg_test}, nt={nt}:")
        print(Diff)
        assert False, "New and old implementations do not match within tolerance"
    
    
def test_I_H():
    for pol_deg_trial in [1, 2, 3, 4, 5]:
        for pol_deg_test in [1, 2, 3, 4, 5]:
            for nt in [100]:
                _test_I_H(pol_deg_trial, pol_deg_test, nt)
                
                
def test_dt_H():
    for pol_deg_trial in [1, 2, 3, 4, 5]:
        for pol_deg_test in [1, 2, 3, 4, 5]:
            for nt in [100]:
                _test_dt_H(pol_deg_trial, pol_deg_test, nt)