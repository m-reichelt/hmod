from hmod.transformations import LegendreToHilbertBase


def get_sparsity_pattern_matrix_sine(nmodes: int, nt: int, polynomial_degree: int):
    legendre_to_hilbert = LegendreToHilbertBase(nmodes, nt, polynomial_degree, basis_type='sine')
    E = legendre_to_hilbert.get_compound_extension_matrix()
    J = legendre_to_hilbert.get_compound_filter_matrix()
    K = np.array([k+0.5 for k in range(J.shape[0])])
    K = np.diag(K)
    J2 = J.T @ K @ J
    pat = E.T @ J2 @ E
    return pat


def get_sparsity_pattern_matrix_cosine(nmodes: int, nt: int, polynomial_degree: int):
    legendre_to_hilbert = LegendreToHilbertBase(nmodes, nt, polynomial_degree, basis_type='cosine')
    E = legendre_to_hilbert.get_compound_extension_matrix()
    J = legendre_to_hilbert.get_compound_filter_matrix()
    K = np.array([k+0.5 for k in range(J.shape[0])])
    K = np.diag(K)
    J2 = J.T @ K @ J
    pat = E.T @ J2 @ E
    return pat

def get_sparsity_pattern_matrix_sine_cosine(nmodes: int, nt: int, polynomial_degree: int):
    legendre_to_hilbert_sin = LegendreToHilbertBase(nmodes, nt, polynomial_degree, basis_type='sine')
    legendre_to_hilbert_cos = LegendreToHilbertBase(nmodes, nt, polynomial_degree, basis_type='cosine')
    Js = legendre_to_hilbert_sin.get_compound_filter_matrix()
    Jc = legendre_to_hilbert_cos.get_compound_filter_matrix()
    Es = legendre_to_hilbert_sin.get_compound_extension_matrix()
    Ec = legendre_to_hilbert_cos.get_compound_extension_matrix()
    Ms = Js @ Es
    Mc = Jc @ Ec
    pat = Mc.T @ Ms
    return pat.todense()



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    nmodes = 240
    nt = 40
    polynomial_degree = 4

    pattern_matrix_sine = get_sparsity_pattern_matrix_sine(nmodes, nt, polynomial_degree)
    pattern_matrix_cosine = get_sparsity_pattern_matrix_cosine(nmodes, nt, polynomial_degree)
    pattern_matrix_sine_cosine = get_sparsity_pattern_matrix_sine_cosine(nmodes, nt, polynomial_degree)
    #spy with color indicating entry value
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1)
    plt.title("Sine Basis Sparsity Pattern")
    plt.imshow(pattern_matrix_sine)
    plt.subplot(2, 3, 2)
    plt.title("Cosine Basis Sparsity Pattern")
    plt.imshow(pattern_matrix_cosine)
    plt.subplot(2, 3, 3)
    plt.title("Sine-Cosine Basis Sparsity Pattern")
    plt.imshow(pattern_matrix_sine_cosine)
    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plt.title("Sine Basis Inverse Sparsity Pattern")
    plt.imshow(np.linalg.inv(pattern_matrix_sine))
    plt.subplot(2, 3, 5)
    plt.title("Cosine Basis Inverse Sparsity Pattern")
    plt.imshow(np.linalg.inv(pattern_matrix_cosine))
    plt.subplot(2, 3, 6)
    plt.title("Sine-Cosine Basis Inverse Sparsity Pattern")
    try:
        sc_inv = np.linalg.inv(pattern_matrix_sine_cosine)
    except np.linalg.LinAlgError:
        sc_inv = np.ones(pattern_matrix_sine_cosine.shape)*np.nan

    plt.imshow(np.linalg.inv(pattern_matrix_sine_cosine))
    plt.tight_layout()
    #add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(), ax=plt.gcf().get_axes(), orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Entry Value')
    plt.show()