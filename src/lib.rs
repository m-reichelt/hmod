#![allow(non_snake_case)]
extern crate core;

mod piecewise_polynomials;
mod sparse_matrix_tools;

use nalgebra::{DMatrix, DMatrixView};
use pyo3::prelude::*;
use crate::sparse_matrix_tools::csr_to_triplets;

#[pyclass]
struct LegendreBasis{
    pub degree : usize,
    pub n_t : usize,
    legendre_basis : piecewise_polynomials::LegendreBasis,
}

#[pymethods]
impl LegendreBasis {
    #[new]
    fn new(degree : usize, n_t : usize, T : f64) -> Self {
        let legendre_basis = piecewise_polynomials::LegendreBasis::new(degree, n_t, T);
        Self {degree, n_t, legendre_basis}
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


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn hmod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(lagrange_to_legendre_basis_transformation, m)?)?;
    m.add_function(wrap_pyfunction!(get_lagrange_points, m)?)?;
    m.add_class::<LegendreBasis>()?;
    m.add_class::<LegendreBasisEvaluator>()?;
    Ok(())
}
