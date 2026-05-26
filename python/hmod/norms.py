
def compute_l2_norm(f, nt : int, T : float, quad_tol :float =1e-8):
    """
    Compute the L2 norm of a given function f.

    Parameters:
    f (callable): Function to compute the L2 norm of. Should take a single argument (time).
    nt (int): Number of time intervals.
    T (float): Total time interval [0, T].
    quad_tol (float): Tolerance for numerical on each interval.

    Returns:
    float: L2 norm of the vector.
    """
    import numpy as np
    from scipy.integrate import quad
    integrand = lambda t: f(t)**2
    timepoints = np.linspace(0, T, nt + 1)
    l2_norm_sq = 0.0
    for i in range(nt):
        a = timepoints[i]
        b = timepoints[i + 1]
        integral, _ = quad(integrand, a, b, epsabs=quad_tol)
        l2_norm_sq += integral
    l2_norm = np.sqrt(l2_norm_sq)
    return l2_norm


def compute_h12_seminorm(f, T : float, quad_tol :float =1e-10):
    """
    Compute the H1/2 seminorm of a given function f.

    Parameters:
    f (callable): Function to compute the H1/2 seminorm of. Should take a single argument (time).
    T (float): Total time interval [0, T].
    quad_tol (float): Tolerance for numerical integration.

    Returns:
    float: H1/2 seminorm of the vector.
    """
    import numpy as np
    from scipy.fft import dst

    nt = np.ceil(2*np.sqrt(1./quad_tol)).astype(int)
    timepoints = np.linspace(0, T, nt+1)
    f_samples = f(timepoints)

    n = len(f_samples)
    h = 1.0 / n
    f_hat = dst(f_samples, type=4)
    omegas = np.pi * (np.arange(n) + 0.5)
    seminorm_squared = 0.5 * h * h * np.sum(omegas * f_hat * f_hat)
    return np.sqrt(seminorm_squared)


