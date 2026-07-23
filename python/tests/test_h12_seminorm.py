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


def test_periodic_h12_seminorm_for_fourier_mode_and_constant():
    from hmod.norms import compute_h12_seminorm
    import numpy as np

    T = 2.75
    mode = 3
    sine = lambda t: np.sin(2 * np.pi * mode * t / T)
    constant = lambda t: np.ones_like(t)

    actual = compute_h12_seminorm(
        sine, T=T, quad_tol=1e-6, periodic=True
    )
    expected = np.sqrt(np.pi * mode)

    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert compute_h12_seminorm(
        constant, T=T, quad_tol=1e-6, periodic=True
    ) < 1e-14








if __name__ == "__main__":
    pytest.main([__file__])
