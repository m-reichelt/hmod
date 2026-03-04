use nalgebra_sparse::{CsrMatrix, CooMatrix};
use crate::transformations::degree_dependent_transformations::HilbertBasisType;

fn get_dst_extension_matrix(n_t: usize, K : usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(K, n_t);
    let two_n = 2 * n_t;
    for k in 0..K {
        // we need to calculate the corresponding fourier coefficient and place it in the correct column
        let r = k.rem_euclid(two_n);
        let m = k.div_euclid(two_n);
        let k_star = if r < n_t { r} else { two_n - r - 1};
        let fac = if m %2 ==0 {1.0} else {-1.0};
        coo.push(k, k_star, fac);
    }
    let csr: CsrMatrix<f64> = CsrMatrix::from(&coo);
    csr
}

pub fn get_dct_extension_matrix(n_t: usize, K : usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(K, n_t);
    let two_n = 2 * n_t;
    for k in 0..K {
        // we need to calculate the corresponding fourier coefficient and place it in the correct column
        let r = k.rem_euclid(two_n);
        let m = k.div_euclid(two_n);
        let k_star = if r < n_t { r} else { two_n - r - 1};
        let fac = if m %2 ==0 {1.0} else {-1.0};
        if r < n_t {
            coo.push(k, k_star, fac);
        }
        else {
            coo.push(k, k_star, -fac);
        }
    }
    let csr: CsrMatrix<f64> = CsrMatrix::from(&coo);
    csr
}



pub fn get_compound_extension_matrix(n_t: usize, K : usize, degree : usize, hilbert_basis_type : HilbertBasisType) -> CsrMatrix<f64> {
    use crate::transformations::sparse_matrix_tools::diagonal_block_matrix_from_csr_vector;
    let DST_ext = get_dst_extension_matrix(n_t, K);
    let DCT_ext = get_dct_extension_matrix(n_t, K);
    let diagonals = (0..=degree).map(|m| {
        if m % 2 == 0 {
            match hilbert_basis_type {
                HilbertBasisType::Sine => DST_ext.clone(),
                HilbertBasisType::Cosine => DCT_ext.clone(),
            }
        } else {
            match hilbert_basis_type {
                HilbertBasisType::Sine => DCT_ext.clone(),
                HilbertBasisType::Cosine => DST_ext.clone(),
            }

        }
    }).collect::<Vec<CsrMatrix<f64>>>();
    let mat = diagonal_block_matrix_from_csr_vector(&diagonals);
    mat
}



#[cfg(test)]
mod tests {
    use nalgebra::DVector;
    use super::*;

    #[test]
    fn test_dst_extension_matrix() {
        let n_t = 4;
        let u_k = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let K = 5*n_t;
        let ext_matrix = get_dst_extension_matrix(n_t, K);
        let u_k_ext = ext_matrix * u_k;
        let expected = vec![1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0];
        for (a, b) in u_k_ext.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_dct_extension_matrix() {
        let n_t = 4;
        let u_k = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let K = 5*n_t;
        let ext_matrix = get_dct_extension_matrix(n_t, K);
        let u_k_ext = ext_matrix * u_k;
        let expected = vec![1.0, 2.0, 3.0, 4.0, -4.0, -3.0, -2.0, -1.0, -1.0, -2.0, -3.0, -4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        for (a, b) in u_k_ext.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "a={a}, b={b}");
        }
    }
}