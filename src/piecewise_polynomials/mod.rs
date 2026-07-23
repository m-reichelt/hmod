#![allow(non_snake_case)]

use scirs2_special::legendre;
use nalgebra::{DMatrix, DVector, Dyn, U1};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use rayon::prelude::*;

const PARALLEL_EVALUATION_THRESHOLD: usize = 2048;

pub(crate) struct LegendreBasis{
    degree: usize,
    n_intervals: usize,
    T : f64 // our domain is [0,T]
}

fn find_interval(t: f64, T: f64, n_intervals: usize) -> usize {
    //we assume t_i = i*T/n_intervals
    let h = T/(n_intervals as f64);
    let mut interval_index = (t/h).floor() as usize;
    if interval_index >= n_intervals {
        interval_index = n_intervals - 1;
    }
    interval_index
}

fn evaluate_legendre_polynomial_derivative(t : f64, n : usize) -> f64
{
    if n==0{
        0.
    }
    else if n==1{
        1.
    }
    else {
        let result = (t*legendre(n, t)-legendre(n-1, t))*(n as f64)/(t*t-1.0+1e-15); //add small number to avoid division by zero
        result
    }
}

impl LegendreBasis {
    pub fn new(degree: usize, n_intervals: usize, T: f64) -> Self {
        Self {
            degree,
            n_intervals,
            T,
        }
    }

    pub fn evaluate_for_interval(&self, t: f64, interval_index: usize, dof_vals : &DMatrix<f64>) -> f64 {
        let interval_length = self.T / (self.n_intervals as f64);
        let a = interval_index as f64 * interval_length;
        let b = a + interval_length;
        let t_ref = transform_from_interval_to_unit_interval(t, (a, b));
        let interval_dof_vals = dof_vals.row(interval_index);
        let result = (0..=self.degree).map(|j| {
            let Pj = legendre(j, t_ref);
            interval_dof_vals[j] * Pj
        }).sum();
        result
    }


    pub fn evaluate_derivative_for_interval(&self, t: f64, interval_index: usize, dof_vals : &DMatrix<f64>) -> f64 {
        let interval_length = self.T / (self.n_intervals as f64);
        let a = interval_index as f64 * interval_length;
        let b = a + interval_length;
        let t_ref = transform_from_interval_to_unit_interval(t, (a, b));
        let interval_dof_vals = dof_vals.row(interval_index);
        let result : f64= (0..=self.degree).map(|j| {
            let Pj_deriv = evaluate_legendre_polynomial_derivative(t_ref, j);
            interval_dof_vals[j] * Pj_deriv
        }).sum();
        result*2.0/interval_length //at last multiply with weight according to transformation of derivatives
    }

    pub fn evaluate_all_basis_functions(&self, t: f64) -> DVector<f64> {
        let interval_index = find_interval(t, self.T, self.n_intervals);
        let interval_length = self.T / (self.n_intervals as f64);
        let a = interval_index as f64 * interval_length;
        let b = a + interval_length;
        let t_ref = transform_from_interval_to_unit_interval(t, (a, b));
        let mut basis_vals = DVector::<f64>::zeros(self.degree + 1);
        for j in 0..=self.degree {
            basis_vals[j] = legendre(j, t_ref);
        }
        //get DMatrix with zeros except for the current interval
        let mut result = DMatrix::<f64>::zeros(self.n_intervals, self.degree + 1);
        result.set_row(interval_index, &basis_vals.transpose());
        let n_coeffs_total = result.len();
        //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
        let result = result.reshape_generic(Dyn(n_coeffs_total), U1);
        result
    }

    pub fn evaluate_all_basis_function_derivatives(&self, t: f64) -> DVector<f64> {
        let interval_index = find_interval(t, self.T, self.n_intervals);
        let interval_length = self.T / (self.n_intervals as f64);
        let a = interval_index as f64 * interval_length;
        let b = a + interval_length;
        let t_ref = transform_from_interval_to_unit_interval(t, (a, b));
        let mut basis_vals = DVector::<f64>::zeros(self.degree + 1);
        for j in 0..=self.degree {
            basis_vals[j] = evaluate_legendre_polynomial_derivative(t_ref, j)*2.0/interval_length;
        }
        //get DMatrix with zeros except for the current interval
        let mut result = DMatrix::<f64>::zeros(self.n_intervals, self.degree + 1);
        result.set_row(interval_index, &basis_vals.transpose());
        let n_coeffs_total = result.len();
        //now we need to reshape from (n_t, degree+1) to (nt*(degree+1)), with column major ordering
        let result = result.reshape_generic(Dyn(n_coeffs_total), U1);
        result
    }

    pub fn evaluate(&self, t: f64, dof_vals : &DMatrix<f64>) -> f64 {
        let interval_index = find_interval(t, self.T, self.n_intervals);
        self.evaluate_for_interval(t, interval_index, dof_vals)
    }

    pub fn evaluate_derivative(&self, t: f64, dof_vals : &DMatrix<f64>) -> f64 {
        let interval_index = find_interval(t, self.T, self.n_intervals);
        self.evaluate_derivative_for_interval(t, interval_index, dof_vals)
    }

    pub fn evaluate_many(&self, t_values: &[f64], dof_vals : &DMatrix<f64>) -> Vec<f64> {
        if t_values.len() >= PARALLEL_EVALUATION_THRESHOLD {
            t_values.par_iter().map(|&t| self.evaluate(t, dof_vals)).collect()
        } else {
            t_values.iter().map(|&t| self.evaluate(t, dof_vals)).collect()
        }
    }

    pub fn evaluate_derivative_many(&self, t_values: &[f64], dof_vals : &DMatrix<f64>) -> Vec<f64> {
        if t_values.len() >= PARALLEL_EVALUATION_THRESHOLD {
            t_values.par_iter().map(|&t| self.evaluate_derivative(t, dof_vals)).collect()
        } else {
            t_values.iter().map(|&t| self.evaluate_derivative(t, dof_vals)).collect()
        }
    }
}


/// Converts Lagrange basis functions on the unit interval [-1, 1] to Legendre basis functions on the same interval.
/// M[i,j] gives the coefficient of the j-th Legendre basis function in the i-th Lagrange basis function.
/// For now all is done for equidistant points. Later we can add support for other points (e.g., Chebyshev).
fn lagrange_to_legendre_unit_interval(degree: usize) -> DMatrix<f64> {
    let n_dofs = degree + 1;
    let mut M = DMatrix::<f64>::zeros(n_dofs, n_dofs);
    match degree {
        0 => {
            panic!("Lagrange to Legendre conversion not meaningful for degree 0");
        },
        1 => {
            M[(0,0)] =  0.5;
            M[(0,1)] = -0.5;
            M[(1,0)] =  0.5;
            M[(1,1)] =  0.5;
        },
        2 => {
             M[(0,0)] =  1.0/6.0; M[(0,1)] = -0.5;      M[(0,2)] =  1.0/3.0;
             M[(1,0)] =  2.0/3.0; M[(1,1)] =  0.0;      M[(1,2)] = -2.0/3.0;
             M[(2,0)] =  1.0/6.0; M[(2,1)] =  0.5;      M[(2,2)] =  1.0/3.0;
        },
        _ => {
            //here we use the Evaluation matrix (Vandermonde like) to compute the conversion matrix
            let evaluation_points = equidistant_points_on_interval(degree, (-1.0, 1.0));
            let V_legendre = DMatrix::<f64>::from_fn(n_dofs, n_dofs, |i, j| {
                legendre(i, evaluation_points[j])
            });
            M = V_legendre.try_inverse().expect("Failed to invert Vandermonde matrix for Lagrange to Legendre conversion.");
        }
    };
    M
}


/// Transform from reference interval [-1, 1] to arbitrary interval [a, b]
#[cfg(test)]
fn transform_from_unit_interval_to_interval(
    t_ref : f64,
    interval : (f64, f64)
) -> f64 {
    let (a, b) = interval;
    let result = 0.5 * ( (b - a) * t_ref + (a + b) );
    let result = if result < a {
        a
    } else if result > b {
        b
    } else {
        result
    };
    result
}

/// Transform from arbitrary interval [a, b] to reference interval [-1, 1]
fn transform_from_interval_to_unit_interval(
    t : f64,
    interval : (f64, f64)
) -> f64 {
    let (a, b) = interval;
    let result = 2.0 * ( t - 0.5 * (a + b) ) / (b - a);
    let result = if result < -1.0 {
        -1.0
    } else if result > 1.0 {
        1.0
    } else {
        result
    };
    result
}

pub(crate) struct LagrangeBasis{
    degree: usize,
    n_intervals: usize,
    periodic: bool,
    // Global nodal DOFs. The periodic topology omits T because it is
    // identified with the first point at t=0.
    lagrange_points : Vec<f64>,
}

fn equidistant_points_on_interval(degree: usize, interval: (f64, f64)) -> Vec<f64> {
    let (a, b) = interval;
    (0..=degree).map(|i| a + (b - a) * (i as f64) / (degree as f64)).collect()
}

fn barycentric_weights(points: &[f64]) -> Vec<f64> {
    points.iter().enumerate().map(|(j, &xj)| {
        let product: f64 = points.iter().enumerate()
            .filter(|&(m, _)| m != j)
            .map(|(_, &xm)| xj - xm)
            .product();
        1.0 / product
    }).collect()
}

fn evaluate_lagrange_basis_at(t: f64, interpolation_points: &[f64], weights: &[f64], values: &mut [f64]) {
    values.fill(0.0);
    for (j, &xj) in interpolation_points.iter().enumerate() {
        if (t - xj).abs() < 1e-14 {
            values[j] = 1.0;
            return;
        }
    }

    let mut denominator = 0.0;
    for (j, &xj) in interpolation_points.iter().enumerate() {
        values[j] = weights[j] / (t - xj);
        denominator += values[j];
    }
    for j in 0..interpolation_points.len() {
        values[j] /= denominator;
    }
}


impl LagrangeBasis {
    pub fn new(degree: usize, n_intervals: usize, T: f64, periodic: bool) -> Self {
        assert!(
            !periodic || degree > 0,
            "Periodic Lagrange bases require polynomial degree at least 1"
        );
        assert!(
            !periodic || n_intervals > 0,
            "Periodic Lagrange bases require at least one interval"
        );
        let interval_length = T / (n_intervals as f64);
        let intervals: Vec<(f64, f64)> = (0..n_intervals)
            .map(|i| {
                let a = i as f64 * interval_length;
                let b = a + interval_length;
                (a, b)
            })
            .collect();
        let mut lagrange_points: Vec<f64> = intervals.iter()
            .flat_map(|&interval| {
                // at some point replace with more stable points (e.g., Chebyshev)
                let mut inter_points = equidistant_points_on_interval(degree, interval);
                // Remove the last point to avoid duplication
                inter_points.pop();
                inter_points
            })
            .collect();
        // For a non-periodic basis, the endpoint at T is an independent DOF.
        // For a periodic basis it is identified with the first DOF at t=0.
        if !periodic {
            lagrange_points.push(T);
        }
        Self {
            degree,
            n_intervals,
            periodic,
            lagrange_points,
        }
    }

    pub fn get_dofs_for_interval(&self, interval_index: usize) -> Vec<usize> {
        let start = interval_index*self.degree;
        let end = start + self.degree+1;
        if self.periodic {
            let n_dofs = self.lagrange_points.len();
            (start..end).map(|dof| dof % n_dofs).collect()
        } else {
            (start..end).collect()
        }
    }

    pub fn get_lagrange_points(&self) -> &Vec<f64> {
        &self.lagrange_points
    }

    /// Convert Lagrange basis function values to Legendre basis function values on each interval.
    /// Input: lagrange_vals contains the values of the Lagrange basis functions
    /// at their respective DOFs.
    /// Output: A matrix of size (n_intervals, degree+1) where each row contains
    /// the Legendre basis function values for that interval.
    #[cfg(test)]
    pub fn to_legendre_basis_vals(&self, lagrange_vals: &DVector<f64>) -> DMatrix<f64> {
        assert_eq!(lagrange_vals.nrows(), self.lagrange_points.len(), "Input lagrange_vals length does not match expected number of DOFs.");
        let M_unit_interval = lagrange_to_legendre_unit_interval(self.degree);
        let mut legendre_vals = DMatrix::zeros(self.n_intervals, self.degree+1);
        for l in 0..self.n_intervals {
            let dof_indices = self.get_dofs_for_interval(l);
            let lagrange_vals_interval = DVector::from_iterator(
                dof_indices.len(),
                dof_indices.iter().map(|&dof| lagrange_vals[dof]),
            );
            let cur_legendre_vals = lagrange_vals_interval.transpose()*&M_unit_interval;
            legendre_vals.set_row(l, &cur_legendre_vals);
        }
        legendre_vals
    }

    /// mainly for testing purposes, applies the same as to_legendre_basis_vals using the csr matrix
    #[cfg(test)]
    pub fn to_legendre_basis_vals_by_matrix(&self, lagrange_vals: &DVector<f64>) -> DMatrix<f64> {
        assert_eq!(lagrange_vals.nrows(), self.lagrange_points.len(), "Input lagrange_vals length does not match expected number of DOFs.");
        let mut legendre_vals = DMatrix::zeros(self.n_intervals, self.degree+1);
        for m in 0..=self.degree {
            let Tm = self.to_legendre_basis_matrix_for_pol_degree(m);
            let legendre_vals_col = Tm*lagrange_vals;
            legendre_vals.set_column(m, &legendre_vals_col);
        };
        legendre_vals
    }

    /// get the transformation matrix to legendre dofs (stacked, i.e [d0; d1; ...])
    pub fn get_transormation_matrix_to_legendre(&self) -> CsrMatrix<f64> {
        use crate::sparse_matrix_tools::csr_block_from_ndarray;
        use ndarray::{ Array2};
        let transformation_matrices : Vec<CsrMatrix<f64>> = (0..=self.degree).map(|d| self.to_legendre_basis_matrix_for_pol_degree(d)).collect();
        let mat_array = Array2::from_shape_fn((self.degree+1,1), |(i, _)| &transformation_matrices[i]);
        let T = csr_block_from_ndarray(&mat_array);
        T
    }


    /// This routine returns the matrix, that does the same as to_legendre_basis_vals, for one specific polynomial degree (column).
    pub fn to_legendre_basis_matrix_for_pol_degree(&self, pol_degree : usize) -> CsrMatrix<f64> {
        let mut Tm = CooMatrix::new(self.n_intervals, self.lagrange_points.len());
        let M_unit_interval = lagrange_to_legendre_unit_interval(self.degree);
        let M_m_row = M_unit_interval.column(pol_degree);
        for l in 0..self.n_intervals {
            let dof_indices = self.get_dofs_for_interval(l);
            for (local_dof, global_dof) in dof_indices.into_iter().enumerate() {
                Tm.push(l, global_dof, M_m_row[local_dof]);
            }
        }
        // now convert to csr
        let T_csr  = CsrMatrix::from(&Tm);
        T_csr
    }

    pub fn get_prolongation_matrix_to(&self, n_intervals_fine: usize) -> CsrMatrix<f64> {
        assert!(self.degree > 0, "Lagrange prolongation requires polynomial degree at least 1");
        assert!(self.n_intervals > 0, "Lagrange prolongation requires at least one coarse interval");
        assert!(n_intervals_fine > 0, "Lagrange prolongation requires at least one fine interval");

        let n_fine_segments = n_intervals_fine*self.degree;
        let n_rows = if self.periodic {
            n_fine_segments
        } else {
            n_fine_segments + 1
        };
        let n_cols = self.lagrange_points.len();
        let interpolation_points = equidistant_points_on_interval(self.degree, (-1.0, 1.0));
        let weights = barycentric_weights(&interpolation_points);
        let mut local_values = vec![0.0; self.degree + 1];
        let mut P = CooMatrix::new(n_rows, n_cols);

        for row in 0..n_rows {
            let t = (row as f64) / (n_fine_segments as f64);
            let mut coarse_interval = (t*(self.n_intervals as f64)).floor() as usize;
            if coarse_interval >= self.n_intervals {
                coarse_interval = self.n_intervals - 1;
            }
            let t_ref = 2.0*(t*(self.n_intervals as f64) - (coarse_interval as f64)) - 1.0;
            evaluate_lagrange_basis_at(t_ref, &interpolation_points, &weights, &mut local_values);
            let col_start = coarse_interval*self.degree;
            for (local_col, &value) in local_values.iter().enumerate() {
                if value != 0.0 {
                    let mut global_col = col_start + local_col;
                    if self.periodic {
                        global_col %= n_cols;
                    }
                    P.push(row, global_col, value);
                }
            }
        }

        CsrMatrix::from(&P)
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    fn csr_to_dense(matrix: &CsrMatrix<f64>) -> DMatrix<f64> {
        let mut result = DMatrix::zeros(matrix.nrows(), matrix.ncols());
        for row in 0..matrix.nrows() {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                result[(row, matrix.col_indices()[entry])] += matrix.values()[entry];
            }
        }
        result
    }

    #[test]
    fn test_legendre_basis_functions() {
        let P0 = |_x: f64| 1.;
        let P1 = |x: f64| x;
        let P2 = |x: f64| 0.5 * (3. * x * x - 1.);
        let xvals = [-1.0, -0.5, 0.0, 0.5, 1.0];
        for &x in &xvals {
            let pl0 = legendre(0, x);
            let pl1 = legendre(1, x);
            let pl2 = legendre(2, x);
            assert!((pl0 - P0(x)).abs() < 1e-10, "P0 mismatch at x={}: {} vs {}", x, pl0, P0(x));
            assert!((pl1 - P1(x)).abs() < 1e-10, "P1 mismatch at x={}: {} vs {}", x, pl1, P1(x));
            assert!((pl2 - P2(x)).abs() < 1e-10, "P2 mismatch at x={}: {} vs {}", x, pl2, P2(x));
        }
    }

    #[test]
    fn test_lagrange_basis_dof_indices_p1(){
        let degree = 1;
        let n_intervals = 3;
        let T = 1.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let dofs_interval_0 = lagrange_basis.get_dofs_for_interval(0);
        let dofs_interval_1 = lagrange_basis.get_dofs_for_interval(1);
        let dofs_interval_2 = lagrange_basis.get_dofs_for_interval(2);
        assert_eq!(dofs_interval_0, vec![0, 1]);
        assert_eq!(dofs_interval_1, vec![1, 2]);
        assert_eq!(dofs_interval_2, vec![2, 3]);
    }
    #[test]
    fn test_lagrange_basis_dof_indices_p2(){
        let degree = 2;
        let n_intervals = 3;
        let T = 1.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let dofs_interval_0 = lagrange_basis.get_dofs_for_interval(0);
        let dofs_interval_1 = lagrange_basis.get_dofs_for_interval(1);
        let dofs_interval_2 = lagrange_basis.get_dofs_for_interval(2);
        assert_eq!(dofs_interval_0, vec![0, 1, 2]);
        assert_eq!(dofs_interval_1, vec![2, 3, 4]);
        assert_eq!(dofs_interval_2, vec![4, 5, 6]);
    }

    #[test]
    fn test_lagrange_basis_dof_indices_p3(){
        let degree = 3;
        let n_intervals = 3;
        let T = 1.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let dofs_interval_0 = lagrange_basis.get_dofs_for_interval(0);
        let dofs_interval_1 = lagrange_basis.get_dofs_for_interval(1);
        let dofs_interval_2 = lagrange_basis.get_dofs_for_interval(2);
        assert_eq!(dofs_interval_0, vec![0, 1, 2, 3]);
        assert_eq!(dofs_interval_1, vec![3, 4, 5, 6]);
        assert_eq!(dofs_interval_2, vec![6, 7, 8, 9]);
    }

    #[test]
    fn test_periodic_lagrange_points_and_wrapped_dofs() {
        let lagrange_basis = LagrangeBasis::new(2, 3, 3.0, true);

        assert_eq!(
            lagrange_basis.get_lagrange_points(),
            &vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        );
        assert_eq!(lagrange_basis.get_dofs_for_interval(0), vec![0, 1, 2]);
        assert_eq!(lagrange_basis.get_dofs_for_interval(1), vec![2, 3, 4]);
        assert_eq!(lagrange_basis.get_dofs_for_interval(2), vec![4, 5, 0]);
    }

    #[test]
    fn test_periodic_lagrange_to_legendre_p1() {
        let lagrange_basis = LagrangeBasis::new(1, 3, 1.0, true);
        let transformation = lagrange_basis.get_transormation_matrix_to_legendre();
        let transformation_dense = csr_to_dense(&transformation);
        let expected = DMatrix::from_row_slice(
            6,
            3,
            &[
                 0.5,  0.5,  0.0,
                 0.0,  0.5,  0.5,
                 0.5,  0.0,  0.5,
                -0.5,  0.5,  0.0,
                 0.0, -0.5,  0.5,
                 0.5,  0.0, -0.5,
            ],
        );

        assert_eq!(transformation.nrows(), 6);
        assert_eq!(transformation.ncols(), 3);
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert!(
                    (transformation_dense[(row, col)] - expected[(row, col)]).abs() < 1e-14,
                    "Periodic transformation mismatch at ({row}, {col})"
                );
            }
        }

        let lagrange_values = DVector::from_vec(vec![2.0, 4.0, 8.0]);
        let expected_coefficients = DVector::from_vec(vec![3.0, 6.0, 5.0, 1.0, 2.0, -3.0]);
        let matrix_coefficients = &transformation * &lagrange_values;
        for row in 0..expected_coefficients.len() {
            assert!(
                (matrix_coefficients[row] - expected_coefficients[row]).abs() < 1e-14
            );
        }

        let direct_coefficients = lagrange_basis.to_legendre_basis_vals(&lagrange_values);
        let matrix_coefficients_by_degree =
            lagrange_basis.to_legendre_basis_vals_by_matrix(&lagrange_values);
        for interval in 0..lagrange_basis.n_intervals {
            for degree in 0..=lagrange_basis.degree {
                assert!(
                    (direct_coefficients[(interval, degree)]
                        - matrix_coefficients_by_degree[(interval, degree)])
                        .abs()
                        < 1e-14
                );
            }
        }
    }

    #[test]
    fn test_periodic_single_interval_identifies_both_endpoints() {
        let lagrange_basis = LagrangeBasis::new(1, 1, 1.0, true);
        let transformation =
            csr_to_dense(&lagrange_basis.get_transormation_matrix_to_legendre());

        assert_eq!(transformation.nrows(), 2);
        assert_eq!(transformation.ncols(), 1);
        assert!((transformation[(0, 0)] - 1.0).abs() < 1e-14);
        assert!(transformation[(1, 0)].abs() < 1e-14);
    }

    #[test]
    fn test_transformations(){
        let interval = (2.0, 4.0);
        let t = 3.5;
        let t_ref = transform_from_interval_to_unit_interval(t, interval);
        //check that t_ref is correct
        assert!((t_ref - 0.5).abs() < 1e-10, "Transformation to reference interval failed: t_ref = {}", t_ref);
        let t_back = transform_from_unit_interval_to_interval(t_ref, interval);
        assert!((t - t_back).abs() < 1e-10, "Transformation roundtrip failed: t = {}, t_back = {}", t, t_back);
    }

    #[test]
    fn test_equidistant_points_p1(){
        let degree = 1;
        let interval = (0.0, 1.0);
        let points = equidistant_points_on_interval(degree, interval);
        assert_eq!(points, vec![0.0, 1.0]);
    }
    #[test]
    fn test_equidistant_points_p2(){
        let degree = 2;
        let interval = (0.0, 1.0);
        let points = equidistant_points_on_interval(degree, interval);
        assert_eq!(points, vec![0.0, 0.5,  1.0]);
    }

    #[test]
    fn test_lagrange_poitns_p1(){
        let degree = 1;
        let n_intervals = 3;
        let T = 3.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let points = lagrange_basis.get_lagrange_points();
        assert_eq!(points, &vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lagrange_poitns_p2(){
        let degree = 2;
        let n_intervals = 2;
        let T = 2.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let points = lagrange_basis.get_lagrange_points();
        assert_eq!(points, &vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_lagrange_prolongation_p1(){
        let lagrange_basis = LagrangeBasis::new(1, 2, 1.0, false);
        let prolongation = lagrange_basis.get_prolongation_matrix_to(4);
        let rows = prolongation.row_offsets();
        let cols = prolongation.col_indices();
        let vals = prolongation.values();

        let expected = vec![
            vec![(0, 1.0)],
            vec![(0, 0.5), (1, 0.5)],
            vec![(1, 1.0)],
            vec![(1, 0.5), (2, 0.5)],
            vec![(2, 1.0)],
        ];

        assert_eq!(prolongation.nrows(), 5);
        assert_eq!(prolongation.ncols(), 3);
        for row in 0..prolongation.nrows() {
            let start = rows[row];
            let end = rows[row + 1];
            assert_eq!(end - start, expected[row].len());
            for (entry, &(expected_col, expected_val)) in expected[row].iter().enumerate() {
                assert_eq!(cols[start + entry], expected_col);
                assert!((vals[start + entry] - expected_val).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_periodic_lagrange_prolongation_p1() {
        let lagrange_basis = LagrangeBasis::new(1, 2, 1.0, true);
        let prolongation = lagrange_basis.get_prolongation_matrix_to(4);
        let prolongation_dense = csr_to_dense(&prolongation);
        let expected = DMatrix::from_row_slice(
            4,
            2,
            &[
                1.0, 0.0,
                0.5, 0.5,
                0.0, 1.0,
                0.5, 0.5,
            ],
        );

        assert_eq!(prolongation.nrows(), 4);
        assert_eq!(prolongation.ncols(), 2);
        for row in 0..expected.nrows() {
            for col in 0..expected.ncols() {
                assert!(
                    (prolongation_dense[(row, col)] - expected[(row, col)]).abs() < 1e-14,
                    "Periodic prolongation mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn test_lagrange_evaluation_p1(){
        use ndarray::linspace;
        let degree = 1;
        let fun = |x: f64| 2.0 * x + 1.0;
        let n_intervals = 15;
        let T = 4.5;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let lagrange_points = lagrange_basis.get_lagrange_points();
        let lagrange_vals: DVector<f64> = DVector::from_iterator(
            lagrange_points.len(),
            lagrange_points.iter().map(|&x| fun(x)),
        );
        let legendre_vals = lagrange_basis.to_legendre_basis_vals(&lagrange_vals);
        {
            //do the same using the csr matrix
            let legendre_vals_by_matrix = lagrange_basis.to_legendre_basis_vals_by_matrix(&lagrange_vals);
            for i in 0..legendre_vals.nrows() {
                for j in 0..legendre_vals.ncols() {
                    assert!((legendre_vals[(i, j)] - legendre_vals_by_matrix[(i, j)]).abs() < 1e-12, "Mismatch in legendre vals at ({},{}) : {} vs {}", i, j, legendre_vals[(i, j)], legendre_vals_by_matrix[(i, j)]);
                }
            }
        }
        let sample_points : Vec<f64>= linspace(0.0, T, 100).collect();
        //construct legendre basis
        let legendre_basis = LegendreBasis::new(degree, n_intervals, T);
        for t in sample_points {
            let interpolated_val = legendre_basis.evaluate(t, &legendre_vals);
            let true_val = fun(t);
            assert!((interpolated_val - true_val).abs() < 1e-10, "Interpolation failed at t={}: {} vs {}", t, interpolated_val, true_val);
        }
    }
    #[test]
    fn test_lagrange_evaluation_p2(){
        use ndarray::linspace;
        let degree = 2;
        let fun = |x: f64| 2.0 * x*x -5.0*x + 1.0;
        let n_intervals = 13;
        let T = 4.5;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let lagrange_points = lagrange_basis.get_lagrange_points();
        let lagrange_vals: DVector<f64> = DVector::from_iterator(
            lagrange_points.len(),
            lagrange_points.iter().map(|&x| fun(x)),
        );
        let legendre_vals = lagrange_basis.to_legendre_basis_vals(&lagrange_vals);
        {
            //do the same using the csr matrix
            let legendre_vals_by_matrix = lagrange_basis.to_legendre_basis_vals_by_matrix(&lagrange_vals);
            for i in 0..legendre_vals.nrows() {
                for j in 0..legendre_vals.ncols() {
                    assert!((legendre_vals[(i, j)] - legendre_vals_by_matrix[(i, j)]).abs() < 1e-12, "Mismatch in legendre vals at ({},{}) : {} vs {}", i, j, legendre_vals[(i, j)], legendre_vals_by_matrix[(i, j)]);
                }
            }
        }
        let sample_points : Vec<f64>= linspace(0.0, T, 100).collect();
        //construct legendre basis
        let legendre_basis = LegendreBasis::new(degree, n_intervals, T);
        for t in sample_points {
            let interpolated_val = legendre_basis.evaluate(t, &legendre_vals);
            let true_val = fun(t);
            assert!((interpolated_val - true_val).abs() < 1e-10, "Interpolation failed at t={}: {} vs {}", t, interpolated_val, true_val);
        }
    }

    #[test]
    fn test_lagrange_evaluation_p3(){
        use ndarray::linspace;
        let degree = 3;
        let fun = |x: f64| 2.0 * x*x*x+15.0*x*x -5.0*x + 1.0;
        let n_intervals = 14;
        let T = 4.5;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T, false);
        let lagrange_points = lagrange_basis.get_lagrange_points();
        let lagrange_vals: DVector<f64> = DVector::from_iterator(
            lagrange_points.len(),
            lagrange_points.iter().map(|&x| fun(x)),
        );
        let legendre_vals = lagrange_basis.to_legendre_basis_vals(&lagrange_vals);
        {
            //do the same using the csr matrix
            let legendre_vals_by_matrix = lagrange_basis.to_legendre_basis_vals_by_matrix(&lagrange_vals);
            for i in 0..legendre_vals.nrows() {
                for j in 0..legendre_vals.ncols() {
                    assert!((legendre_vals[(i, j)] - legendre_vals_by_matrix[(i, j)]).abs() < 1e-12, "Mismatch in legendre vals at ({},{}) : {} vs {}", i, j, legendre_vals[(i, j)], legendre_vals_by_matrix[(i, j)]);
                }
            }
        }
        let sample_points : Vec<f64>= linspace(0.0, T, 100).collect();
        //construct legendre basis
        let legendre_basis = LegendreBasis::new(degree, n_intervals, T);
        for t in sample_points {
            let interpolated_val = legendre_basis.evaluate(t, &legendre_vals);
            let true_val = fun(t);
            assert!((interpolated_val - true_val).abs() < 1e-10, "Interpolation failed at t={}: {} vs {}", t, interpolated_val, true_val);
        }
    }


}
