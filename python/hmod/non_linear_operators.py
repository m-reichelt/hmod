from enum import Enum
import hmod.standard_matrices as sm
import hmod.hilbert_matrices as hm
import numpy as np
import scipy.sparse as sp
#import pypardiso as ppd

class ResidualType(Enum):
    Standard = 1
    Hilbert = 2


class WeightedResidual:
    """Return standard- or Hilbert-tested residuals in a Lagrange test space.

    With ``periodic=True``, the test space identifies the endpoints and the
    Hilbert variant uses the periodic modified Hilbert transform.
    """
    def __init__(
        self,
        residual_type: ResidualType,
        polynomial_degree_test: int,
        polynomial_degree_projection: int,
        nt: int,
        T: float,
        periodic: bool = False,
    ):
        self.residual_type = residual_type
        self.polynomial_degree_test = polynomial_degree_test
        self.polynomial_degree_projection = polynomial_degree_projection
        self.nt = nt
        self.T = T
        self.periodic = periodic
        #define the projection matrix onto Legendre polynomials
        self.M_L = sm.get_legendre_legendre_matrix_for_derivatives(
            polynomial_degree_projection,
            polynomial_degree_projection,
            0,
            0,
            nt,
            T,
            periodic=periodic,
        )
        #perform a sparse LU decomposition of M_L
        self.M_L_LU = sp.linalg.splu(self.M_L.tocsc())
        self.n_modes = int(1e5)
        if residual_type == ResidualType.Standard:
            self.Mass = sm.get_legendre_lagrange_matrix_for_derivatives(
                polynomial_degree_projection,
                polynomial_degree_test,
                0,
                0,
                nt,
                T,
                periodic=periodic,
            )
        elif residual_type == ResidualType.Hilbert:
            self.Mass = hm.get_hilbert_matrix_for_derivatives_legendre_lagrange(
                polynomial_degree_projection,
                polynomial_degree_test,
                0,
                0,
                nt,
                T,
                periodic=periodic,
            )
    def project_fun(self, residual_fun):
        res_vec_rhs = sm.rhs_quadrature(
            residual_fun,
            self.nt,
            self.polynomial_degree_projection,
            self.T,
            periodic=self.periodic,
        )
        res_vec = self.M_L_LU.solve(res_vec_rhs)
        #res_vec = ppd.spsolve(self.M_L, res_vec_rhs)
        return res_vec

    def apply(self, residual_fun):
        res_vec = self.project_fun(residual_fun)
        y = self.Mass @ res_vec
        return y
