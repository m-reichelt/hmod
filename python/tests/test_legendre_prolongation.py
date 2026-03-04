import pytest
import numpy as np



def _test_prolongation_for_degree(poly_deg : int):
    nt = 23
    T = 1
    from hmod.polynomial_bases import legendre_refinement_matrix, LegendreBasisEvaluator
    #get the prolongation matrix
    P = legendre_refinement_matrix(poly_deg, nt)
    Pd = np.array(P.todense())
    #check size
    expected_rows = (2*(poly_deg + 1))*nt
    expected_cols = (poly_deg + 1)*nt
    assert P.shape == (expected_rows, expected_cols), f"Prolongation matrix has incorrect shape {P.shape}, expected {(expected_rows, expected_cols)}"
    #get a vector of random coefficients on coarse level with fixed seed
    np.random.seed(42)
    dofs_coarse = np.random.rand((poly_deg + 1)*nt)
    #dofs_coarse = np.array([1.,2.])
    #evaluate on fine level via prolongation
    dof_fine_via_P = P @ dofs_coarse
    #get evaluators for both levels
    legendre_evaluator_coarse = LegendreBasisEvaluator(dofs_coarse, poly_deg, nt, T)
    legendre_evaluator_fine = LegendreBasisEvaluator(dof_fine_via_P, poly_deg, 2*nt, T)
    t_test = np.linspace(0, T, 100)
    vals_coarse = legendre_evaluator_coarse.evaluate(t_test)
    vals_fine = legendre_evaluator_fine.evaluate(t_test)
    diff = np.linalg.norm(vals_coarse - vals_fine)
    tol = 1e-12
    assert diff < tol, f"Prolongation test failed for polynomial degree {poly_deg} with difference {diff}"




def test_prolongation():
    degrees_to_test = [0,1,2,3,4,5]
    for deg in degrees_to_test:
        _test_prolongation_for_degree(deg)



if __name__ == "__main__":
    pytest.main([__file__])