import pytest
import numpy as np
import hmod.standard_matrices as sm
import hmod.hilbert_matrices as hm
from hmod.preconditioning import LowestOrderPreconditioner
import hmod.matrix_tools as mat_tools

def test_lowest_order_preconditioner():
    nt = 15
    T = 1.0
    mu = 3.8
    nmodes = int(1e5)
    preconditioner = LowestOrderPreconditioner(nt, mu, T)

    #get the norm inducing matrix
    At = hm.Operator_dt_H_Lagrange_Lagrange(nmodes, nt, 1, 1)
    #transform to dense
    At = mat_tools.linear_operator_to_matrix(At)
    Mt = sm.get_lagrange_lagrange_matrix_for_derivatives(1,1,0,0,nt,T)
    D_full = At + mu*Mt.todense()
    D_full = np.array(D_full)
    #homogenize
    D = D_full[1:,1:]

    #get the inverse via FFT
    D_inv_fft = LowestOrderPreconditioner(nt, mu, T, nmodes)

    # Create input vector
    x = np.arange(nt, dtype=np.float64)+1.0

    # Apply D
    Dx = D @ x
    # Apply preconditioner
    DinvDx = preconditioner @ Dx

    test = DinvDx/x

    diff = DinvDx - x
    error_norm = np.linalg.norm(diff)

    assert error_norm < 1e-8, f"Preconditioner test failed with error norm {error_norm}"






if __name__ == "__main__":
    pytest.main([__file__])