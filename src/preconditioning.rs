use pyo3::prelude::*;
use crate::transformations::fft::{FFTType, FFTWwrapper};
use rayon::prelude::*;

#[pyclass]
pub struct EigenBasisTransformLowestOrder{
    n_t : usize,
    T : f64,
    forward_transformation : FFTWwrapper, // DST-II (at the end we need a factor)
    inverse_transformation : FFTWwrapper, // DST-III (at the end we need a factor)
}

#[pymethods]
impl EigenBasisTransformLowestOrder{
    #[new]
    pub fn new(n_t : usize, T : f64) -> Self{
        let forward_transformation = FFTWwrapper::new(n_t, FFTType::DST2);
        let inverse_transformation = FFTWwrapper::new(n_t, FFTType::DST3);
        Self {
            n_t,
            T,
            forward_transformation,
            inverse_transformation,
        }
    }

    pub fn forward_transform(&self, input : Vec<f64>) -> PyResult<Vec<f64>>{
        if input.len() != self.n_t {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("EigenBasisTransformLowestOrder: forward_transform: input length {} does not match n_t {}", input.len(), self.n_t)));
        }
        let res = self.forward_transformation.forward(&input);
        static FAC : f64 = 0.5;
        let res_scaled : Vec<f64> = res.iter().map(|x| x * FAC).collect();
        Ok(res_scaled)
    }

    pub fn inverse_transform(&self, input : Vec<f64>) -> PyResult<Vec<f64>> {
        if input.len() != self.n_t {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("EigenBasisTransformLowestOrder: inverse_transform: input length {} does not match n_t {}", input.len(), self.n_t)));
        }
        let res = self.inverse_transformation.forward(&input);
        let fac = 1./(self.n_t as f64);
        let res_scaled : Vec<f64> = res.iter().map(|x| x * fac).collect();
        Ok(res_scaled)
    }

    fn get_generalized_eigenvalue(&self, l : usize, n_terms : usize) -> f64 {
        use std::f64::consts::PI;
        let xl = (0.5+(l as f64)) * PI / ((2*self.n_t) as f64);
        let fac1 = 1.5*PI/self.T;
        let fac2 = (xl.sin()/xl).powi(4);
        let fac3 = 1./(2.0+(2.0*xl).cos());
        let fac4 = (2.0*l as f64 +1.0).powi(4);
        let fac = fac1 * fac2 * fac3 * fac4;
        let sequence_lambda = |k| 1./((4*k*self.n_t+2*l+1) as f64).powi(3) + 1./((4*k*self.n_t+4*self.n_t-1-2*l) as f64).powi(3);
        let sequence = (0..n_terms).map(sequence_lambda);
        //reverse order of summation for better convergence
        let sum : f64 = sequence.rev().sum();
        fac * sum
    }
    pub fn get_generalized_eigenvalues(&self, n_terms : usize) -> Vec<f64> {
        let generalized_eigenvalues : Vec<f64> = (0..self.n_t).into_par_iter().map(|l| self.get_generalized_eigenvalue(l, n_terms)).collect();
        generalized_eigenvalues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigen_basis_transform_lowest_order() {
        let n_t = 5;
        let T = 1.0;
        let transformer = EigenBasisTransformLowestOrder::new(n_t, T);
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let forward_res = transformer.forward_transform(input.clone()).expect("Forward transform failed");
        let inverse_res = transformer.inverse_transform(forward_res).expect("Inverse transform failed");
        for (a, b) in input.iter().zip(inverse_res.iter()) {
            assert!((a - b).abs() < 1e-9, "a={a}, b={b}");
        }
    }
}