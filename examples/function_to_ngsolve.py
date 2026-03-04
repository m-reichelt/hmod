from hmod.ngsolve_tools import *

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

    gf, fes = lagrange_vector_to_ngsolve_gf(dof_vals, nt, degree, T)

    # Plot the GridFunction
    tplot = np.linspace(0, T, 100)
    plt.plot(tplot, gf(tplot), label='Hat function in NGSolve GF')
    plt.legend()
    plt.show()