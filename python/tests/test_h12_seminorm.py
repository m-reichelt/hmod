import pytest


def test_h12_norm():
    from hmod.norms import compute_h12_seminorm
    import numpy as np
    k = 4
    f_analytic = lambda t: np.sin((k +0.5) * np.pi * t)
    T = 1.0
    h12_semi = compute_h12_seminorm(f_analytic, T=T, quad_tol=1e-8)
    exact_semi = np.sqrt(0.5 * (k +0.5) * np.pi)
    diff = np.abs(h12_semi - exact_semi)
    assert diff < 1e-6








if __name__ == "__main__":
    pytest.main([__file__])