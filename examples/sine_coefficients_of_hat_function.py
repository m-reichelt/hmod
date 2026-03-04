from hmod.transformations import LegendreToHilbertBase
from hmod.polynomial_bases import LegendreBasis, get_lagrange_to_legendre_matrix






def evaluate_sine_coefficients(t, sine_coeffs, T):
    base_function = lambda t, k : np.sin((k + 0.5) * np.pi * t / T)
    f_approx = np.zeros_like(t)
    for k, coeff in enumerate(sine_coeffs):
        f_approx += coeff * base_function(t, k)
    return f_approx




if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the linearly spaced interval
    nt = 2
    degree = 1
    T = 2.
    t = np.linspace(0, T, nt + 1)

    # Define the dof index of the hat function
    dof_index = 1
    # Set the vectors of the dof vals
    dof_vals = np.zeros(nt + 1)
    dof_vals[dof_index] = 1.

    #get the first K sine base coefficients
    K = 400
    Trans = LegendreToHilbertBase(K, nt, degree, basis_type='sine')
    #get the lagrange to legendre transformation matrix
    B = get_lagrange_to_legendre_matrix(degree, nt)

    #evaluate the sine base approximation of the hat function
    tplot = np.linspace(0, T, 40+1)
    legendre_dofs = B@dof_vals
    coeffs = Trans@legendre_dofs
    c2 = Trans@(B@dof_vals)

    from hmod.hilbert_matrices import Operator_dt_H_Lagrange_Lagrange
    Op_dtH = Operator_dt_H_Lagrange_Lagrange(nmodes=K, nt=nt, polynomial_degree_trial=degree, polynomial_degree_test=degree)

    f_approx = evaluate_sine_coefficients(tplot, coeffs, T)

    #plot the hat function
    plt.figure()
    let_base = LegendreBasis(degree, nt, T)
    plt.plot(tplot, let_base.evaluate(tplot, legendre_dofs), label='Hat function')
    plt.plot(tplot, f_approx, label='Sine base approximation')
    plt.legend()
    plt.show()