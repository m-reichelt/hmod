import numpy as np
import hmod.hilbert_matrices as hm
import hmod.deprecated_hilbert_matrices as hmd

nt = 200
polynomial_degree = 1
def make_old(n):
    A_old = hmd.Operator_dt_H_Lagrange_Lagrange(int(1e5), n, polynomial_degree, polynomial_degree)
    return A_old

def make_new(n):
    A_new = hm.get_hilbert_matrix_with_derivatives_lagrange_lagrange(n, polynomial_degree, polynomial_degree, 1, 0,
                                                                     1.)
    return A_new

def test_construct_old(benchmark):
    benchmark(make_old, nt)

def test_construct_new(benchmark):
    benchmark(make_new, nt)

def test_matvec_old(benchmark):
    A = make_old(nt)
    x = np.random.default_rng(0).standard_normal(A.shape[1])
    benchmark(A.matvec, x)   # or benchmark(lambda: A @ x)

def test_matvec_new(benchmark):
    A = make_new(nt)
    x = np.random.default_rng(0).standard_normal(A.shape[1])
    benchmark(A.matvec, x)

def test_matmat_old(benchmark):
    A = make_old(nt)
    X = np.random.default_rng(0).standard_normal((A.shape[1], 16))
    benchmark(A.matmat, X)   # or benchmark(lambda: A @ X)

def test_matmat_new(benchmark):
    A = make_new(nt)
    X = np.random.default_rng(0).standard_normal((A.shape[1], 16))
    benchmark(A.matmat, X)