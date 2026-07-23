
def compute_l2_norm(
    f,
    nt: int,
    T: float,
    quad_tol: float = 1e-8,
    periodic: bool = False,
):
    """
    Compute the L2 norm of a given function f.

    Parameters:
    f (callable): Function to compute the L2 norm of. Should take a single argument (time).
    nt (int): Number of time intervals.
    T (float): Total time interval [0, T].
    quad_tol (float): Tolerance for numerical on each interval.
    periodic (bool): Accepted for a common periodic/non-periodic API.

    Returns:
    float: L2 norm of the vector.

    The integral is topology-independent; ``periodic`` is accepted so the
    same norm API can be used for periodic and non-periodic examples.
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


def compute_h12_seminorm(
    f,
    T: float,
    quad_tol: float = 1e-10,
    periodic: bool = False,
):
    """
    Compute the H1/2 seminorm of a given function f.

    Parameters:
    f (callable): Function to compute the H1/2 seminorm of. Should take a single argument (time).
    T (float): Total time interval [0, T].
    quad_tol (float): Tolerance for numerical integration.
    periodic (bool): Use the periodic Fourier seminorm when true.

    Returns:
    float: H1/2 seminorm of the vector. With ``periodic=True``, this uses
    the periodic Fourier definition
    ``T * sum_k |omega_k| * |f_hat_k|**2``.
    """
    import numpy as np

    if periodic:
        # With c_k = (1/T) int_0^T f(t) exp(-i*omega_k*t) dt,
        # |f|^2_{H^(1/2)_per} = T sum_k |omega_k| |c_k|^2.
        n_samples = max(16, int(np.ceil(2 / np.sqrt(quad_tol))))
        timepoints = np.linspace(0.0, T, n_samples, endpoint=False)
        samples = np.asarray(f(timepoints))
        if samples.shape == ():
            samples = np.full(n_samples, samples)
        samples = np.broadcast_to(samples, (n_samples,))
        coefficients = np.fft.fft(samples) / n_samples
        frequencies = 2 * np.pi * np.fft.fftfreq(
            n_samples, d=T / n_samples
        )
        seminorm_squared = T * np.sum(
            np.abs(frequencies) * np.abs(coefficients) ** 2
        )
        return np.sqrt(max(float(np.real(seminorm_squared)), 0.0))

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
