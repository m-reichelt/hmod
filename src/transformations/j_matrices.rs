use scirs2_special::spherical_jn;
use nalgebra_sparse::{CsrMatrix, CooMatrix};


fn get_jm_tilde_matrix(k_max : usize, n_t : usize, m : usize) -> CsrMatrix<f64> {
    let h_t = 1.0 / (n_t as f64);
    let omega_k = |k: usize| (k as f64 + 0.5) * std::f64::consts::PI;
    let diagonal_entries = (0..k_max).map(|k| {
        let omega = omega_k(k);
        spherical_jn(m as i32, omega * h_t*0.5)
    });
    let mut coo = CooMatrix::new(k_max, k_max);
    for (i, value) in diagonal_entries.enumerate() {
        coo.push(i, i, value);
    }
    let csr: CsrMatrix<f64> = CsrMatrix::from(&coo);
    csr
}

pub fn get_compound_filter(k_max : usize, n_t : usize, degree : usize) -> CsrMatrix<f64> {
    use ndarray::{ Array2};
    use crate::transformations::sparse_matrix_tools::csr_block_from_ndarray;
    let mut blocks: Vec<CsrMatrix<f64>> = Vec::new();
    for m in 0..=degree {
        let block = get_jm_tilde_matrix(k_max, n_t, m);
        blocks.push(block);
    }
    let arr = Array2::from_shape_fn((1, degree + 1), |(_, j)| {
        &blocks[j]
    });
    let mat = csr_block_from_ndarray(&arr);
    mat
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::linspace;

    #[test]
    fn test_bessel_jl() {
        let j0_expected = |t : f64| t.sin() / t;
        let j1_expected = |t : f64| (t.sin() / (t*t)) - (t.cos() / t);
        let j2_expected = |t : f64| ((3.0/(t*t) - 1.0) * (t.sin()/t)) - (3.0 * t.cos() / (t*t));
        let ts = linspace(0.1, 20.0, 100).collect::<Vec<_>>();
        for &t in ts.iter() {
            let j0 = spherical_jn(0, t);
            let j1 = spherical_jn(1, t);
            let j2 = spherical_jn(2, t);
            let j0_exp = j0_expected(t);
            let j1_exp = j1_expected(t);
            let j2_exp = j2_expected(t);
            let tol = 1e-6;
            assert!((j0 - j0_exp).abs() < tol, "j0({}) = {}, expected {}", t, j0, j0_exp);
            assert!((j1 - j1_exp).abs() < tol, "j1({}) = {}, expected {}", t, j1, j1_exp);
            assert!((j2 - j2_exp).abs() < tol, "j2({}) = {}, expected {}", t, j2, j2_exp);
        }
    }
}