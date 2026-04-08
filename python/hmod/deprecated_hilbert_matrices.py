import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator


class Base_Operator_Sine_Sine_Legendre(LinearOperator):
    """ Base class for operator with using sine base on both trial and test side."""
    def __init__(self, fourier_factors : np.ndarray, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        from hmod.transformations import LegendreToHilbertBase
        self.nmodes = nmodes
        self.nt = nt
        #get the class containing the transformation for trial functions
        self.TrialTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_trial, basis_type='sine') #base transform on the right
        #get the class containing the transformation for test functions
        self.TestTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_test, basis_type='sine') #base transform on the left
        #get the two extension matrices
        self.ExtensionTrial = self.TrialTransform.get_compound_extension_matrix() #extension for trial
        self.ExtensionTest = self.TestTransform.get_compound_extension_matrix() #extension for test
        self.ExtensionTest_T = self.ExtensionTest.transpose()
        #get the two filter matrices
        self.FilterTrial = self.TrialTransform.get_compound_filter_matrix() #filter for trial
        self.FilterTest = self.TestTransform.get_compound_filter_matrix() #filter for test
        #fourier factor times 0.5 for the integral
        F = scipy.sparse.diags(fourier_factors) *0.5
        #F = scipy.sparse.identity(nmodes)*0.5
        #put together the kernel matrix
        self.K = self.ExtensionTest_T @ self.FilterTest.transpose() @ F @ self.FilterTrial @ self.ExtensionTrial #todo: sum up the operators better
        n_rows = nt*(polynomial_degree_test+1)
        n_cols = nt*(polynomial_degree_trial+1)
        shape = (n_rows, n_cols)
        super().__init__(dtype=None, shape=shape)


    def _matvec(self, x):
        #x_legendre = self.Ttrial @ x
        x_legendre = x
        x_f = self.TrialTransform.apply_fft(x_legendre)
        y_f = self.K @ x_f
        y_legendre = self.TestTransform.apply_fft_transpose(y_f)
        #y = self.Ttest.transpose() @ y_legendre
        y = y_legendre
        fac = (1.0/self.nt)**2
        return y*fac

    def _adjoint(self, x):
        #x_legendre = self.Ttest @ x
        x_legendre = x
        x_f = self.TestTransform.apply_fft(x_legendre)
        y_f = self.K.transpose() @ x_f
        y_legendre = self.TrialTransform.apply_fft_transpose(y_f)
        #y = self.Ttrial.transpose() @ y_legendre
        y = y_legendre
        fac = (1.0/self.nt)**2
        return y*fac


class Base_Operator_Sine_Sine_Lagrange_Lagrange(LinearOperator):
    def __init__(self, fourier_factors : np.ndarray, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        from hmod.polynomial_bases import get_lagrange_to_legendre_matrix
        #to legendre basis for trial and test functions
        self.Ttrial = get_lagrange_to_legendre_matrix(polynomial_degree_trial, nt) #base transform on the right (to legendre basis)
        self.Ttest = get_lagrange_to_legendre_matrix(polynomial_degree_test, nt) #base transform on the left (to legendre basis)
        self.legendre_op = Base_Operator_Sine_Sine_Legendre(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)
        nrows = self.Ttest.shape[1]
        ncols = self.Ttrial.shape[1]
        shape = (nrows, ncols)
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        x_legendre = self.Ttrial @ x
        y_legendre = self.legendre_op @ x_legendre
        y = self.Ttest.transpose() @ y_legendre
        return y

    def _adjoint(self, x):
        x_legendre = self.Ttest @ x
        y_legendre = self.legendre_op.transpose() @ x_legendre
        y = self.Ttrial.transpose() @ y_legendre
        return y



class Base_Operator_Cosine_Sine_Legendre(LinearOperator):
    """ Base class for operator with using cosine base on trial side and sine base on test side."""
    def __init__(self, fourier_factors : np.ndarray, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        from hmod.transformations import LegendreToHilbertBase
        self.nmodes = nmodes
        self.nt = nt
        #get the class containing the transformation for trial functions
        self.TrialTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_trial, basis_type='cosine') #base transform on the right
        #get the class containing the transformation for test functions
        self.TestTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_test, basis_type='sine') #base transform on the left
        #get the two extension matrices
        self.ExtensionTrial = self.TrialTransform.get_compound_extension_matrix() #extension for trial
        self.ExtensionTest = self.TestTransform.get_compound_extension_matrix() #extension for test
        self.ExtensionTest_T = self.ExtensionTest.transpose()
        #get the two filter matrices
        self.FilterTrial = self.TrialTransform.get_compound_filter_matrix() #filter for trial
        self.FilterTest = self.TestTransform.get_compound_filter_matrix() #filter for test
        #fourier factor times 0.5 for the integral
        F = scipy.sparse.diags(fourier_factors) *0.5
        #put together the kernel matrix
        self.K = self.ExtensionTest_T @ self.FilterTest.transpose() @ F @ self.FilterTrial @ self.ExtensionTrial #todo: sum up the operators better

        n_rows = nt*(polynomial_degree_test+1)
        n_cols = nt*(polynomial_degree_trial+1)
        shape = (n_rows, n_cols)
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        #x_legendre = self.Ttrial @ x
        x_legendre = x
        x_f = self.TrialTransform.apply_fft(x_legendre)
        y_f = self.K @ x_f
        y_legendre = self.TestTransform.apply_fft_transpose(y_f)
        #y = self.Ttest.transpose() @ y_legendre
        y = y_legendre
        fac = (1.0/self.nt)**2
        return y*fac

    def _adjoint(self, x):
        #x_legendre = self.Ttest @ x
        x_legendre = x
        x_f = self.TestTransform.apply_fft(x_legendre)
        y_f = self.K.transpose() @ x_f
        y_legendre = self.TrialTransform.apply_fft_transpose(y_f)
        #y = self.Ttrial.transpose() @ y_legendre
        y = y_legendre
        fac = (1.0/self.nt)**2
        return y*fac

class Base_Operator_Cosine_Sine_Lagrange_Lagrange(LinearOperator):
    def __init__(self, fourier_factors : np.ndarray, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        from hmod.polynomial_bases import get_lagrange_to_legendre_matrix
        #to legendre basis for trial and test functions
        self.Ttrial = get_lagrange_to_legendre_matrix(polynomial_degree_trial, nt) #base transform on the right (to legendre basis)
        self.Ttest = get_lagrange_to_legendre_matrix(polynomial_degree_test, nt) #base transform on the left (to legendre basis)
        self.legendre_op = Base_Operator_Cosine_Sine_Legendre(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)
        nrows = self.Ttest.shape[1]
        ncols = self.Ttrial.shape[1]
        shape = (nrows, ncols)
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        x_legendre = self.Ttrial @ x
        y_legendre = self.legendre_op @ x_legendre
        y = self.Ttest.transpose() @ y_legendre
        return y

    def _adjoint(self, x):
        x_legendre = self.Ttest @ x
        y_legendre = self.legendre_op.transpose() @ x_legendre
        y = self.Ttrial.transpose() @ y_legendre
        return y

class Base_Operator_Cosine_Sine_Legendre_Lagrange(LinearOperator):
    def __init__(self, fourier_factors : np.ndarray, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        from hmod.polynomial_bases import get_lagrange_to_legendre_matrix
        #to legendre basis for test functions
        self.Ttest = get_lagrange_to_legendre_matrix(polynomial_degree_test, nt) #base transform on the left (to legendre basis)
        self.legendre_op = Base_Operator_Cosine_Sine_Legendre(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)
        nrows = self.Ttest.shape[1]
        ncols = nt*(polynomial_degree_trial+1)
        shape = (nrows, ncols)
        super().__init__(dtype=None, shape=shape)

    def _matvec(self, x):
        x_legendre = x
        y_legendre = self.legendre_op @ x_legendre
        y = self.Ttest.transpose() @ y_legendre
        return y

    def _adjoint(self, x):
        x_legendre = self.Ttest @ x
        y_legendre = self.legendre_op.transpose() @ x_legendre
        y = y_legendre
        return y

class Operator_dt_H_Lagrange_Lagrange(Base_Operator_Sine_Sine_Lagrange_Lagrange):
    def __init__(self, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        fourier_factors = np.array([(0.5+k)*np.pi for k in range(nmodes)])
        super().__init__(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)


class Operator_I_H_Legendre_Lagrange(Base_Operator_Cosine_Sine_Legendre_Lagrange):
    def __init__(self, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        fourier_factors = np.array([1. for k in range(nmodes)])
        super().__init__(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)


class Operator_I_H_Lagrange_Lagrange(Base_Operator_Cosine_Sine_Lagrange_Lagrange):
    def __init__(self, nmodes: int, nt : int, polynomial_degree_trial : int, polynomial_degree_test : int):
        fourier_factors = np.array([1. for k in range(nmodes)])
        super().__init__(fourier_factors, nmodes, nt, polynomial_degree_trial, polynomial_degree_test)


class Rhs_I_H_Lagrange_Lagrange(LinearOperator):
    def __init__(self, nmodes: int, nt : int, polynomial_degree_rhs : int, polynomial_degree_test : int):
        from hmod.transformations import LegendreToHilbertBase
        from hmod.polynomial_bases import get_lagrange_to_legendre_matrix
        self.nmodes = nmodes
        self.nt = nt
        #to legendre basis for trial function
        self.Ttest = get_lagrange_to_legendre_matrix(polynomial_degree_test, nt) #base transform on the right (to legendre basis)
        #get the class containing the transformation for trial functions
        self.BaseTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_test, basis_type='sine') #base transform on the right
        #get the class containing the transformation for rhs functions
        self.RhsTransform = LegendreToHilbertBase(nmodes, nt, polynomial_degree_rhs, basis_type='cosine') #base transform on the right
        #get the two extension matrices
        self.ExtensionRhs = self.RhsTransform.get_compound_extension_matrix() #extension for rhs
        self.ExtensionTest = self.BaseTransform.get_compound_extension_matrix() #extension for test
        self.ExtensionTest_T = self.ExtensionTest.transpose()
        #get the two filter matrices
        self.FilterRhs = self.RhsTransform.get_compound_filter_matrix() #filter for rhs
        self.FilterTest = self.BaseTransform.get_compound_filter_matrix() #filter for test
        #fourier factor is the identity
        I = scipy.sparse.identity(nmodes) * 0.5 #factor 0.5 for the integral
        #put together the kernel matrix
        self.K = self.ExtensionTest_T @ self.FilterTest.transpose() @ I @ self.FilterRhs @ self.ExtensionRhs #todo: sum up the operators better

        n_rows = self.Ttest.shape[1]
        n_cols = nt*(polynomial_degree_rhs+1)

        shape = (n_rows, n_cols)

        super().__init__(dtype=None, shape=shape)

    def _matvec(self, rhs_legendre_dofs : np.ndarray):
        r_f = self.RhsTransform.apply_fft(rhs_legendre_dofs)
        y_f = self.K @ r_f
        rhs_legendre = self.BaseTransform.apply_fft_transpose(y_f)
        rhs = self.Ttest.transpose() @ rhs_legendre
        fac = (1.0/self.nt)**2
        return rhs*fac