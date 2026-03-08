from scipy.sparse.linalg import LinearOperator
import numpy as np

class LegendreToHilbertBase(LinearOperator):
    def __init__(self, nmodes : int, nt : int, polynomial_degree : int, basis_type : str):
        from hmod.hmod import LegendreToHilbertBasis
        self.nmodes = nmodes
        self.nt= nt
        self.polynomial_degree = polynomial_degree
        self._legendre_to_hilbert = LegendreToHilbertBasis(nmodes, nt, polynomial_degree, basis_type=basis_type)
        shape = (self.nmodes, nt*(polynomial_degree+1))
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        result = np.array(self._legendre_to_hilbert.apply(x))
        return result

    def apply_fft(self, legendre_coeffs : np.ndarray) -> np.ndarray:
        #make sure, that we have in fact an 0-dimensional array
        legendre_coeffs = np.squeeze(legendre_coeffs)
        return np.array(self._legendre_to_hilbert.apply_fft(legendre_coeffs))

    def apply_fft_transpose(self, hilbert_coeffs : np.ndarray) -> np.ndarray:
        #make sure, that we have in fact an 0-dimensional array
        hilbert_coeffs = np.squeeze(hilbert_coeffs)
        return np.array(self._legendre_to_hilbert.apply_ftt_transpose(hilbert_coeffs))

    def get_compound_extension_matrix(self):
        from hmod.matrix_tools import triplets_to_linear_operator
        ri, ci, vals = self._legendre_to_hilbert.get_extension_matrix()
        return triplets_to_linear_operator(ri, ci, vals)

    def get_compound_filter_matrix(self):
        from hmod.matrix_tools import triplets_to_linear_operator
        ri, ci, vals = self._legendre_to_hilbert.get_compound_filter_matrix()
        return triplets_to_linear_operator(ri, ci, vals)