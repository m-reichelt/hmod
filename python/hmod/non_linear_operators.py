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
    """ This class gives as result <residual_fun, \varphi_j> or <residual_fun, H\varphi_j>"""
    def __init__(self, residual_type: ResidualType, polynomial_degree_test : int, polynomial_degree_projection : int, nt : int, T : float):
        self.residual_type = residual_type
        self.polynomial_degree_test = polynomial_degree_test
        self.polynomial_degree_projection = polynomial_degree_projection
        self.nt = nt
        self.T = T
        #define the projection matrix onto Legendre polynomials
        self.M_L = sm.get_legendre_legendre_matrix_for_derivatives(polynomial_degree_projection, polynomial_degree_projection, 0, 0, nt, T)
        #perform a sparse LU decomposition of M_L
        #self.M_L_LU = sp.linalg.splu(self.M_L)
        self.n_modes = int(1e5)
        if residual_type == ResidualType.Standard:
            self.Mass = sm.get_legendre_lagrange_matrix_for_derivatives(polynomial_degree_projection, polynomial_degree_test, 0, 0, nt, T)
        elif residual_type == ResidualType.Hilbert:
            self.Mass = hm.Operator_I_H_Legendre_Lagrange(self.n_modes, nt, polynomial_degree_projection, polynomial_degree_test)
    def project_fun(self, residual_fun):
        res_vec_rhs = sm.rhs_quadrature(residual_fun, self.nt, self.polynomial_degree_projection, self.T)
        res_vec = self.M_L_LU.solve(res_vec_rhs)
        #res_vec = ppd.spsolve(self.M_L, res_vec_rhs)
        return res_vec

    def apply(self, residual_fun):
        res_vec = self.project_fun(residual_fun)
        y = self.Mass @ res_vec
        return y
