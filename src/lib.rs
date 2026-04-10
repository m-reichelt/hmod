#![allow(non_snake_case)]
extern crate core;

mod transformations;
mod piecewise_polynomials;
mod norms;
mod preconditioning;
mod hilbert_kernel;

use nalgebra::{DMatrix, DMatrixView, DVector, Dyn, U1};
use nalgebra_sparse::CsrMatrix;
use pyo3::prelude::*;
use transformations::degree_dependent_transformations::HilbertBasisTransformForSeparateDegrees;
use crate::preconditioning::EigenBasisTransformLowestOrder;
use crate::transformations::degree_dependent_transformations::HilbertBasisType;
use crate::transformations::sparse_matrix_tools::csr_to_triplets;

#[pyclass]
#[derive(Clone)]
struct LegendreToHilbertBasis{
    n_modes : usize,
    n_t : usize,
    degree : usize,
    degree_based_transform: HilbertBasisTransformForSeparateDegrees,
    extension_matrix : CsrMatrix<f64>,
    compund_filter_sine : CsrMatrix<f64>,
    basis_type : HilbertBasisType,
}

#[pymethods]
impl LegendreToHilbertBasis {

    #[new]
    fn new(n_modes : usize, n_t : usize, degree : usize, basis_type : String) -> Self {
        let basis_type = match basis_type.as_str() {
            "sine" => { HilbertBasisType::Sine },
            "cosine" => { HilbertBasisType::Cosine },
            _ => panic!("LegendreToHilbertBasis: only 'sine' or 'cosine' basis types are supported"),
        };
        let degree_based_transform = HilbertBasisTransformForSeparateDegrees::new(n_t, degree, basis_type);
        let extension_matrix = transformations::extension_matrices::get_compound_extension_matrix(n_t, n_modes, degree, basis_type);
        let compund_filter_sine = transformations::j_matrices::get_compound_filter(n_modes, n_t, degree);
        Self {n_modes, n_t, degree, degree_based_transform, extension_matrix, compund_filter_sine, basis_type}
    }

    /// this routine assumes [d_0, d_1, ...], where d_i are the coefficients of i-th degree
    pub fn apply(&self, legendre_dofs : Vec<f64>) -> PyResult<Vec<f64>> {
        let view: DMatrixView<f64> = DMatrixView::from_slice(&legendre_dofs, self.n_t, self.degree+1);
        let hilbert_base_coeffs = self.degree_based_transform.forward(&view);
        let n_coeffs_total = hilbert_base_coeffs.len();
        //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
        let hilbert_base_coeffs_vec = hilbert_base_coeffs.reshape_generic(Dyn(n_coeffs_total), U1);
        //next we need to get the compound extension matrix
        let extended_hilbert_base_coeffs = &self.extension_matrix * &hilbert_base_coeffs_vec;
        // lastly we need to apply the compound filter matrix, that already sums over the degrees
        let filtered_hilbert_base_coeffs = &self.compund_filter_sine * &extended_hilbert_base_coeffs;
        let h = 1. / (self.n_t as f64);
        let filered_hilbert_base_coeffs_scaled = filtered_hilbert_base_coeffs*h;
        let filered_hilbert_base_coeffs_scaled_vec = filered_hilbert_base_coeffs_scaled.as_slice().to_vec();
        Ok(filered_hilbert_base_coeffs_scaled_vec)
    }

    pub fn apply_fft(&self, legendre_dofs : Vec<f64>) -> PyResult<Vec<f64>> {
        let view: DMatrixView<f64> = DMatrixView::from_slice(&legendre_dofs, self.n_t, self.degree+1);
        let hilbert_base_coeffs = self.degree_based_transform.forward(&view);
        let n_coeffs_total = hilbert_base_coeffs.len();
        //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
        let hilbert_base_coeffs_vec = hilbert_base_coeffs.reshape_generic(Dyn(n_coeffs_total), U1);
        let result_vec = hilbert_base_coeffs_vec.as_slice().to_vec();
        Ok(result_vec)
    }

    pub fn apply_ftt_transpose(&self, fft_dofs : Vec<f64>) -> PyResult<Vec<f64>> {
        let view: DMatrixView<f64> = DMatrixView::from_slice(&fft_dofs, self.n_t, self.degree+1);
        let legendre_base_coeffs = self.degree_based_transform.forward(&view);
        let n_coeffs_total = legendre_base_coeffs.len();
        //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
        let legendre_base_coeffs_vec = legendre_base_coeffs.reshape_generic(Dyn(n_coeffs_total), U1);
        let result_vec = legendre_base_coeffs_vec.as_slice().to_vec();
        Ok(result_vec)
    }

    pub fn get_extension_matrix(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        csr_to_triplets(&self.extension_matrix)
    }


    pub fn get_compound_filter_matrix(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        csr_to_triplets(&self.compund_filter_sine)
    }
}

#[pyclass]
struct LegendreBasis{
    pub degree : usize,
    pub n_t : usize,
    T : f64,
    legendre_basis : piecewise_polynomials::LegendreBasis,
}

#[pymethods]
impl LegendreBasis {
    #[new]
    fn new(degree : usize, n_t : usize, T : f64) -> Self {
        let legendre_basis = piecewise_polynomials::LegendreBasis::new(degree, n_t, T);
        Self {degree, n_t, T, legendre_basis}
    }

    fn evaluate_at(&self, t: f64, legendre_vals : Vec<f64>) -> PyResult<f64> {
        //reshape the vector
        let legendre_vals : DMatrix<f64> = DMatrixView::from_slice(&legendre_vals, self.n_t, self.degree+1).try_into().unwrap();
        let val = self.legendre_basis.evaluate(t, &legendre_vals);
        Ok(val)
    }

    pub fn evaluate_all_basis_functions(&self, t: f64) -> PyResult<Vec<f64>> {
        let vals = self.legendre_basis.evaluate_all_basis_functions(t);
        Ok(vals.iter().cloned().collect())
    }

    pub fn evaluate_all_basis_function_derivatives(&self, t: f64) -> PyResult<Vec<f64>> {
        let vals = self.legendre_basis.evaluate_all_basis_function_derivatives(t);
        Ok(vals.iter().cloned().collect())
    }
}


/// Class to evaluate legendre basis more often without copying the dofs from python each time
#[pyclass]
struct LegendreBasisEvaluator{
    legendre_basis : piecewise_polynomials::LegendreBasis,
    n_t : usize,
    degree : usize,
    legendre_vals_vec : Vec<f64>,
}

#[pymethods]
impl LegendreBasisEvaluator {
    #[new]
    fn new(degree : usize, n_t : usize, T : f64) -> Self {
        let legendre_basis = piecewise_polynomials::LegendreBasis::new(degree, n_t, T);
        let legendre_vals_vec = vec![0.0; n_t*(degree+1)];
        Self {legendre_basis, n_t, degree, legendre_vals_vec}
    }

    fn set_dofs(&mut self, legendre_vals : Vec<f64>) -> PyResult<()> {
        assert_eq!(legendre_vals.len(), self.n_t*(self.degree+1), "LegendreBasisEvaluator: set_dofs: input vector has wrong length");
        self.legendre_vals_vec = legendre_vals;
        Ok(())
    }

    fn evaluate_at(&self, t: f64) -> PyResult<f64> {
        //reshape the vector
        let legendre_vals : DMatrix<f64> = DMatrixView::from_slice(&self.legendre_vals_vec, self.n_t, self.degree+1).try_into().unwrap();
        let val = self.legendre_basis.evaluate(t, &legendre_vals);
        Ok(val)
    }

    fn evaluate_derivative_at(&self, t: f64) -> PyResult<f64> {
        //reshape the vector
        let legendre_vals : DMatrix<f64> = DMatrixView::from_slice(&self.legendre_vals_vec, self.n_t, self.degree+1).try_into().unwrap();
        let val = self.legendre_basis.evaluate_derivative(t, &legendre_vals);
        Ok(val)
    }
}

#[pyfunction]
fn lagrange_to_legendre_basis_transformation(degree : usize, n_t : usize) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let T = 1.0; //does not matter for matrix
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, n_t, T);
    let transformation_matrix = lagrange_basis.get_transormation_matrix_to_legendre();
    let (row_indices, col_indices, values) = csr_to_triplets(&transformation_matrix);
    Ok((row_indices, col_indices, values))
}

#[pyfunction]
fn get_lagrange_points(degree : usize, n_t : usize, T : f64) -> PyResult<Vec<f64>> {
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, n_t, T);
    let points = lagrange_basis.get_lagrange_points().clone();
    Ok(points)
}


//compute sine base coefficients of function with lagrange dofs
#[pyfunction]
fn compute_sine_base_coefficients(n_modes : usize, lagrange_dofs: Vec<f64>, n_t : usize, degree : usize, T : f64) -> PyResult<Vec<f64>> {
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, n_t, T);
    let lagrange_vals = DVector::from_vec(lagrange_dofs);
    let legendre_vals = lagrange_basis.to_legendre_basis_vals(&lagrange_vals);
    let legendre_vals = legendre_vals.as_view();
    let mut sine_transform = transformations::degree_dependent_transformations::HilbertBasisTransformForSeparateDegrees::new(n_t, degree, HilbertBasisType::Sine);
    let sine_coeffs = sine_transform.forward(&legendre_vals);
    let n_coeffs_total = sine_coeffs.len();
    //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
    let sine_coeffs_vec = sine_coeffs.reshape_generic(Dyn(n_coeffs_total), U1);
    //next we need to get the compound extension matrix
    let E = transformations::extension_matrices::get_compound_extension_matrix(n_t, n_modes, degree, HilbertBasisType::Sine);
    let extended_sine_coeffs = &E * &sine_coeffs_vec;
    // lastly we need to apply the compound filter matrix, that already sums over the degrees
    let J = transformations::j_matrices::get_compound_filter(n_modes, n_t, degree);
    let filtered_sine_coeffs = &J * &extended_sine_coeffs;
    let h = 1. / (n_t as f64);
    let filered_sine_coeffs_scaled = filtered_sine_coeffs*h;
    Ok(filered_sine_coeffs_scaled.iter().cloned().collect())
}


#[pyfunction]
fn compute_sine_base_coefficients_pw_constants(n_modes : usize, dofs: Vec<f64>, n_t : usize) -> PyResult<Vec<f64>> {
    let degree = 0;
    let legendre_vals = DMatrix::from_fn(dofs.len(), 1, |i, _| dofs[i]);
    let mut sine_transform = transformations::degree_dependent_transformations::HilbertBasisTransformForSeparateDegrees::new(n_t, degree, HilbertBasisType::Sine);
    let legendre_vals = legendre_vals.as_view();
    let sine_coeffs = sine_transform.forward(&legendre_vals);
    let n_coeffs_total = sine_coeffs.len();
    //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
    let sine_coeffs_vec = sine_coeffs.reshape_generic(Dyn(n_coeffs_total), U1);
    //next we need to get the compound extension matrix
    let E = transformations::extension_matrices::get_compound_extension_matrix(n_t, n_modes, degree, HilbertBasisType::Sine);
    let extended_sine_coeffs = &E * &sine_coeffs_vec;
    // lastly we need to apply the compound filter matrix, that already sums over the degrees
    let J = transformations::j_matrices::get_compound_filter(n_modes, n_t, degree);
    let filtered_sine_coeffs = &J * &extended_sine_coeffs;
    let h = 1. / (n_t as f64);
    let filered_sine_coeffs_scaled = filtered_sine_coeffs*h;
    Ok(filered_sine_coeffs_scaled.iter().cloned().collect())
}

/// Get the kernel matrix for modified Hilbert transform
#[pyfunction]
fn get_hilbert_kernel_matrix_for_legendre_degrees(nt : usize, pol_deg_tral : u32, pol_deg_test : u32) -> PyResult<Vec<f64>> {
    use crate::hilbert_kernel::get_kernel_matrix_for_degrees;
    Ok(get_kernel_matrix_for_degrees(nt, pol_deg_tral, pol_deg_test))
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn hmod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sine_base_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sine_base_coefficients_pw_constants, m)?)?;
    m.add_function(wrap_pyfunction!(lagrange_to_legendre_basis_transformation, m)?)?;
    m.add_function(wrap_pyfunction!(get_hilbert_kernel_matrix_for_legendre_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(get_lagrange_points, m)?)?;
    m.add_function(wrap_pyfunction!(norms::h12_seminorm, m)?)?;
    m.add_class::<LegendreToHilbertBasis>()?;
    m.add_class::<LegendreBasis>()?;
    m.add_class::<LegendreBasisEvaluator>()?;
    m.add_class::<EigenBasisTransformLowestOrder>()?;
    Ok(())
}


