#![allow(non_snake_case)]

use scirs2_special::legendre;
use nalgebra::{DMatrix, DVector, Dyn, U1};
use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use crate::transformations::sparse_matrix_tools::csr_block_from_ndarray;

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
    pub fn get_dofs_for_interval(&self, interval_index: usize) -> Vec<usize> {
        let start = interval_index * (self.degree + 1);
        (start..start + self.degree + 1).collect()
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
    intervals : Vec<(f64, f64)>,
    T : f64, // our domain is [0,T]
    lagrange_points : Vec<f64>, // dofs on the interval [0, T]
}

fn equidistant_points_on_interval(degree: usize, interval: (f64, f64)) -> Vec<f64> {
    let (a, b) = interval;
    (0..=degree).map(|i| a + (b - a) * (i as f64) / (degree as f64)).collect()
}


impl LagrangeBasis {
    pub fn new(degree: usize, n_intervals: usize, T: f64) -> Self {
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
        // Add the last point of the last interval
        lagrange_points.push(T);
        Self {
            degree,
            n_intervals,
            intervals,
            T,
            lagrange_points,
        }
    }
    pub fn get_dofs_for_interval(&self, interval_index: usize) -> Vec<usize> {
        let start = interval_index*self.degree;
        let end = start + self.degree+1;
        (start..end).collect()
    }

    pub fn get_lagrange_points(&self) -> &Vec<f64> {
        &self.lagrange_points
    }

    /// Convert Lagrange basis function values to Legendre basis function values on each interval.
    /// Input: lagrange_vals is a vector of length n_intervals * degree +1 containing the values of the Lagrange basis functions at their respective DOFs.
    /// Output: A matrix of size (n_intervals, degree) where each row contains the Legendre basis function values for that interval.
    pub fn to_legendre_basis_vals(&self, lagrange_vals: &DVector<f64>) -> DMatrix<f64> {
        assert_eq!(lagrange_vals.nrows(), self.intervals.len()*self.degree + 1, "Input lagrange_vals length does not match expected number of DOFs.");
        let M_unit_interval = lagrange_to_legendre_unit_interval(self.degree);
        let mut legendre_vals = DMatrix::zeros(self.n_intervals, self.degree+1);
        for l in 0..self.n_intervals {
            let dof_indices = self.get_dofs_for_interval(l);
            let start = dof_indices[0];
            let length = dof_indices.len();
            //get slice of lagrange_vals corresponding to the current interval
            let lagrange_vals_interval = lagrange_vals.rows(start, length);
            let cur_legendre_vals = lagrange_vals_interval.transpose()*&M_unit_interval;
            legendre_vals.set_row(l, &cur_legendre_vals);
        }
        legendre_vals
    }

    /// mainly for testing purposes, applies the same as to_legendre_basis_vals using the csr matrix
    pub fn to_legendre_basis_vals_by_matrix(&self, lagrange_vals: &DVector<f64>) -> DMatrix<f64> {
        assert_eq!(lagrange_vals.nrows(), self.intervals.len()*self.degree + 1, "Input lagrange_vals length does not match expected number of DOFs.");
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
        use crate::transformations::sparse_matrix_tools::csr_block_from_ndarray;
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
            let start = dof_indices[0];
            for d in dof_indices {
                Tm.push(l, d, M_m_row[d - start]);
            }
        }
        // now convert to csr
        let T_csr  = CsrMatrix::from(&Tm);
        T_csr
    }

    /// This routine gives the compound csr matrix to get vectorized polynomial fourier coeffcients
    pub fn get_compound_to_legendre_basis_transform(&self, pol_degree : usize) -> CsrMatrix<f64> {
        use crate::transformations::sparse_matrix_tools::csr_block_from_ndarray;
        use ndarray::{ Array2};
        let transformation_matrices : Vec<CsrMatrix<f64>> = (0..=pol_degree).map(|d| self.to_legendre_basis_matrix_for_pol_degree(d)).collect();
        let mat_array = Array2::from_shape_fn((pol_degree,1), |(i, _)| &transformation_matrices[i]);
        let T = csr_block_from_ndarray(&mat_array);
        T
    }

}


#[cfg(test)]
mod tests {
    use super::*;
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
        let dofs_interval_0 = lagrange_basis.get_dofs_for_interval(0);
        let dofs_interval_1 = lagrange_basis.get_dofs_for_interval(1);
        let dofs_interval_2 = lagrange_basis.get_dofs_for_interval(2);
        assert_eq!(dofs_interval_0, vec![0, 1, 2, 3]);
        assert_eq!(dofs_interval_1, vec![3, 4, 5, 6]);
        assert_eq!(dofs_interval_2, vec![6, 7, 8, 9]);
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
        let points = lagrange_basis.get_lagrange_points();
        assert_eq!(points, &vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_lagrange_poitns_p2(){
        let degree = 2;
        let n_intervals = 2;
        let T = 2.0;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
        let points = lagrange_basis.get_lagrange_points();
        assert_eq!(points, &vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_lagrange_evaluation_p1(){
        use ndarray::linspace;
        let degree = 1;
        let fun = |x: f64| 2.0 * x + 1.0;
        let n_intervals = 15;
        let T = 4.5;
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
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
        let lagrange_basis = LagrangeBasis::new(degree, n_intervals, T);
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