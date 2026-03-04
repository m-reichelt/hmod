import pytest
from hmod.matrix_tools import csr_matvec_reverse
import numpy as np
import scipy.sparse as sp

def test_inverse_csr_mult():
    #get a random sparse matrix in csr format
    np.random.seed(0)
    n = 100
    density = 0.05
    A = sp.random(n, n, density=density, format='csr', dtype=np.float64)
    #get a random vector
    x = np.random.rand(n).astype(np.float64)
    #compute the product using scipy
    y_scipy = A.dot(x)
    #compute the product using our reverse order multiplication
    y_reverse = csr_matvec_reverse(A, x)
    #compare the results
    assert np.allclose(y_scipy, y_reverse, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
