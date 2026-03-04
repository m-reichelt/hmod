from hmod.transformations import LegendreToHilbertBase






def evaluate_cosine_coefficients(t, sine_coeffs, T):
    base_function = lambda t, k : np.cos((k + 0.5) * np.pi * t / T)
    sine_coeffs = [c for c in sine_coeffs]
    f_approx = np.zeros_like(t)
    for k, coeff in enumerate(sine_coeffs):
        f_approx += coeff * base_function(t, k)
    return f_approx




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the linearly spaced interval
    nt = 1
    degree = 0
    T = 1.
    t = np.linspace(0, T, nt + 1)

    # Define the dof index of the hat function
    dof_index = 0
    # Set the vectors of the dof vals
    dof_vals = np.zeros(nt)
    dof_vals[dof_index] = 1.

    #define step function for that dof
    def step_function(x):
        if (t[dof_index] <= x < t[dof_index + 1]):
            return 1.
        else:
            return 0.
    step_function = np.vectorize(step_function)

    #get the first K cosine base coefficients
    K = int(40*nt)

    #get the transformation as Linear operator
    Trans = LegendreToHilbertBase(K, nt, degree, "cosine")
    sine_coeffs_op = Trans@dof_vals

    #evaluate the sine base approximation of the hat function
    tplot = np.linspace(0, T, 100)
    f_approx = evaluate_cosine_coefficients(tplot, sine_coeffs_op, T)

    #plot the hat function
    plt.figure()
    plt.plot(tplot, step_function(tplot), label='Step function')
    plt.plot(tplot, f_approx, label='Cosine base approximation')
    plt.legend()
    plt.show()