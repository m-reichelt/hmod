import numpy as np

def f_analytical(t):
    return np.cos(2 * np.pi * t + 5)


def project_for_refinement(refinement : int, pol_order : int):
    """Get the projection matrix from a coarse grid to a refined grid with given refinement factor and polynomial order."""
    T = 3.0
    nt = 4*2**refinement
    tpoints = np.linspace(0, T, nt+1)
    from hmod.standard_matrices import project_rhs_onto_legendre_basis
    from hmod.polynomial_bases import LegendreBasis
    projected_vals = project_rhs_onto_legendre_basis(f_analytical, nt, pol_order, T=T)
    #get legendre basis and define l2 integrand
    legendre_basis = LegendreBasis(pol_order, nt, T=T)
    l2_integrand = lambda t : (f_analytical(t) - legendre_basis.evaluate(t, projected_vals))**2
    #calculate l2 error
    from scipy.integrate import quad
    l2_error = 0.0
    for ie in range(nt):
        t_left = tpoints[ie]
        t_right = tpoints[ie+1]
        integral, _ = quad(l2_integrand, t_left, t_right)
        l2_error += integral
    l2_error = np.sqrt(l2_error)
    return l2_error

if __name__ == "__main__":
    nlevels = 7
    pol_order = 0
    errL2s = [0.]*nlevels
    eocs = [0.]*nlevels
    for level in range(nlevels):
        errL2s[level] = project_for_refinement(level, pol_order)
        if level > 0:
            eocs[level] = np.log(errL2s[level-1]/errL2s[level]) / np.log(2.0)
        print(f"Level {level}: L2 error = {errL2s[level]:.3e}, EOC = {eocs[level]:.2f}")
