#![allow(non_snake_case)]
extern crate core;

mod piecewise_polynomials;
mod sparse_matrix_tools;

use nalgebra::{DMatrix, DMatrixView};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
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
    legendre_vals : DMatrix<f64>,
}

#[pymethods]
impl LegendreBasisEvaluator {
    #[new]
    fn new(degree : usize, n_t : usize, T : f64) -> Self {
        let legendre_basis = piecewise_polynomials::LegendreBasis::new(degree, n_t, T);
        let legendre_vals = DMatrix::zeros(n_t, degree+1);
        Self {legendre_basis, n_t, degree, legendre_vals}
    }

    fn set_dofs(&mut self, legendre_vals : Vec<f64>) -> PyResult<()> {
        assert_eq!(legendre_vals.len(), self.n_t*(self.degree+1), "LegendreBasisEvaluator: set_dofs: input vector has wrong length");
        self.legendre_vals = DMatrixView::from_slice(&legendre_vals, self.n_t, self.degree+1).try_into().unwrap();
        Ok(())
    }

    fn evaluate_at(&self, t: f64) -> PyResult<f64> {
        let val = self.legendre_basis.evaluate(t, &self.legendre_vals);
        Ok(val)
    }

    fn evaluate_derivative_at(&self, t: f64) -> PyResult<f64> {
        let val = self.legendre_basis.evaluate_derivative(t, &self.legendre_vals);
        Ok(val)
    }

    fn evaluate_many(&self, py: Python<'_>, t_values: PyBuffer<f64>) -> PyResult<Vec<f64>> {
        let t_values = t_values.to_vec(py)?;
        let values = py.detach(|| self.legendre_basis.evaluate_many(&t_values, &self.legendre_vals));
        Ok(values)
    }

    fn evaluate_derivative_many(&self, py: Python<'_>, t_values: PyBuffer<f64>) -> PyResult<Vec<f64>> {
        let t_values = t_values.to_vec(py)?;
        let values = py.detach(|| self.legendre_basis.evaluate_derivative_many(&t_values, &self.legendre_vals));
        Ok(values)
    }
}

#[pyfunction(signature = (degree, n_t, periodic=false))]
fn lagrange_to_legendre_basis_transformation(degree : usize, n_t : usize, periodic : bool) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    if periodic && degree == 0 {
        return Err(PyValueError::new_err(
            "Periodic Lagrange bases require polynomial degree at least 1",
        ));
    }
    if periodic && n_t == 0 {
        return Err(PyValueError::new_err(
            "Periodic Lagrange bases require at least one interval",
        ));
    }
    let T = 1.0; //does not matter for matrix
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, n_t, T, periodic);
    let transformation_matrix = lagrange_basis.get_transormation_matrix_to_legendre();
    let (row_indices, col_indices, values) = csr_to_triplets(&transformation_matrix);
    Ok((row_indices, col_indices, values))
}

#[pyfunction(signature = (nt_coarse, nt_fine, degree, periodic=false))]
fn lagrange_prolongation_matrix(nt_coarse : usize, nt_fine : usize, degree : usize, periodic : bool) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    if periodic && degree == 0 {
        return Err(PyValueError::new_err(
            "Periodic Lagrange bases require polynomial degree at least 1",
        ));
    }
    if periodic && (nt_coarse == 0 || nt_fine == 0) {
        return Err(PyValueError::new_err(
            "Periodic Lagrange prolongation requires positive interval counts",
        ));
    }
    let T = 1.0; //does not matter for matrix
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, nt_coarse, T, periodic);
    let prolongation_matrix = lagrange_basis.get_prolongation_matrix_to(nt_fine);
    let (row_indices, col_indices, values) = csr_to_triplets(&prolongation_matrix);
    Ok((row_indices, col_indices, values))
}

#[pyfunction(signature = (degree, n_t, T, periodic=false))]
fn get_lagrange_points(degree : usize, n_t : usize, T : f64, periodic : bool) -> PyResult<Vec<f64>> {
    if periodic && degree == 0 {
        return Err(PyValueError::new_err(
            "Periodic Lagrange bases require polynomial degree at least 1",
        ));
    }
    if periodic && n_t == 0 {
        return Err(PyValueError::new_err(
            "Periodic Lagrange bases require at least one interval",
        ));
    }
    let lagrange_basis = piecewise_polynomials::LagrangeBasis::new(degree, n_t, T, periodic);
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
    m.add_function(wrap_pyfunction!(lagrange_prolongation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(get_lagrange_points, m)?)?;
    m.add_class::<LegendreBasis>()?;
    m.add_class::<LegendreBasisEvaluator>()?;
    Ok(())
}
