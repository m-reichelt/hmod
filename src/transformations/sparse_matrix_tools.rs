use nalgebra_sparse::{coo::CooMatrix, csr::CsrMatrix};
use ndarray::Array2;

/// Assemble a CSR block matrix from a 2D ndarray of CSR blocks.
/// `blocks[[i, j]]` is the (i, j) block (borrowed).
/// All blocks in block-row i must have equal nrows; all blocks in block-col j equal ncols.
pub fn csr_block_from_ndarray(blocks: &Array2<&CsrMatrix<f64>>) -> CsrMatrix<f64> {
    assert!(blocks.nrows() > 0 && blocks.ncols() > 0, "empty block grid");
    let br = blocks.nrows();
    let bc = blocks.ncols();

    // Per-block-row heights
    let mut row_heights = Vec::with_capacity(br);
    for i in 0..br {
        let h = blocks[[i, 0]].nrows();
        for j in 1..bc {
            assert_eq!(blocks[[i, j]].nrows(), h, "inconsistent block-row height at row {i}");
        }
        row_heights.push(h);
    }
    // Per-block-col widths
    let mut col_widths = Vec::with_capacity(bc);
    for j in 0..bc {
        let w = blocks[[0, j]].ncols();
        for i in 1..br {
            assert_eq!(blocks[[i, j]].ncols(), w, "inconsistent block-col width at col {j}");
        }
        col_widths.push(w);
    }

    // Global size
    let nrows: usize = row_heights.iter().sum();
    let ncols: usize = col_widths.iter().sum();

    // Prefix sums → offsets
    let mut row_offs = Vec::with_capacity(br + 1);
    row_offs.push(0);
    for &h in &row_heights {
        row_offs.push(row_offs.last().unwrap() + h);
    }
    let mut col_offs = Vec::with_capacity(bc + 1);
    col_offs.push(0);
    for &w in &col_widths {
        col_offs.push(col_offs.last().unwrap() + w);
    }

    // Build in COO, then convert to CSR
    let mut coo = CooMatrix::new(nrows, ncols);
    for i in 0..br {
        let r0 = row_offs[i];
        for j in 0..bc {
            let c0 = col_offs[j];
            let blk = blocks[[i, j]];
            let ro = blk.row_offsets();
            let ci = blk.col_indices();
            let va = blk.values();
            for lr in 0..blk.nrows() {
                let start = ro[lr];
                let end = ro[lr + 1];
                let gr = r0 + lr;
                for k in start..end {
                    coo.push(gr, c0 + ci[k], va[k].clone());
                }
            }
        }
    }

    CsrMatrix::from(&coo)
}


pub fn diagonal_block_matrix_from_csr_vector(blocks: &Vec<CsrMatrix<f64>>) -> CsrMatrix<f64> {
    let n = blocks.len();
    let arr = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            blocks[i].clone()
        } else {
            let n_rows = blocks[i].nrows();
            let n_cols = blocks[j].ncols();
            let empty = CooMatrix::new(n_rows, n_cols);
            let empty = CsrMatrix::from(&empty);
            empty
        }
    });
    let array_of_blocks: Array2<&CsrMatrix<f64>> = arr.map(|b| b);
    csr_block_from_ndarray(&array_of_blocks)
}

pub fn csr_to_triplets(a: &CsrMatrix<f64>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let coo = CooMatrix::from(a);
    let ro = coo.row_indices().to_vec();
    let ci = coo.col_indices().to_vec();
    let va = coo.values().to_vec();
    (ro,ci, va)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::csr::CsrMatrix;
    use nalgebra::DMatrix;
    use approx::*;

    /// Build a tiny CSR from triplets (r, c, v).
    fn csr_from_triplets(nrows: usize, ncols: usize, trips: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for &(i, j, v) in trips {
            coo.push(i, j, v);
        }
        CsrMatrix::from(&coo) // From< &CooMatrix > -> CSR
    }

    /// Convert CSR to dense nalgebra::DMatrix (for easy assertions).
    fn csr_to_dense(a: &CsrMatrix<f64>) -> DMatrix<f64> {
        let mut m = DMatrix::<f64>::zeros(a.nrows(), a.ncols());
        let ro = a.row_offsets();
        let ci = a.col_indices();
        let va = a.values();
        for r in 0..a.nrows() {
            for k in ro[r]..ro[r + 1] {
                m[(r, ci[k])] = va[k];
            }
        }
        m
    }

    #[test]
    fn assemble_2x2_block_ndarray() {
        // Build four small, deliberately rectangular blocks:
        // A: 2x3, B: 2x2, C: 1x3, D: 1x2  -> final is (2+1) x (3+2) = 3x5
        let a = csr_from_triplets(2, 3, &[
            (0, 0, 1.0), (0, 2, 2.0),
            (1, 1, 3.0),
        ]);
        let b = csr_from_triplets(2, 2, &[
            (0, 0, 4.0), (1, 1, 5.0),
        ]);
        let c = csr_from_triplets(1, 3, &[
            (0, 1, 6.0),
        ]);
        let d = csr_from_triplets(1, 2, &[
            (0, 0, 7.0), (0, 1, 8.0),
        ]);

        // Put into a 2x2 ndarray of &CsrMatrix
        let blocks: Array2<&CsrMatrix<f64>> = Array2::from_shape_vec(
            (2, 2),
            vec![&a, &b, &c, &d],
        ).unwrap();

        // Assemble
        let big = csr_block_from_ndarray(&blocks);

        // Reference dense: place A B in first two rows, C D in last row
        let mut ref_dense = DMatrix::<f64>::zeros(3, 5);
        // A at (0:2, 0:3)
        ref_dense[(0, 0)] = 1.0; ref_dense[(0, 2)] = 2.0;
        ref_dense[(1, 1)] = 3.0;
        // B at (0:2, 3:5)
        ref_dense[(0, 3)] = 4.0; ref_dense[(1, 4)] = 5.0;
        // C at (2, 0:3)
        ref_dense[(2, 1)] = 6.0;
        // D at (2, 3:5)
        ref_dense[(2, 3)] = 7.0; ref_dense[(2, 4)] = 8.0;

        let big_dense = csr_to_dense(&big);

        // Compare numerically
        assert_eq!(big_dense.nrows(), 3);
        assert_eq!(big_dense.ncols(), 5);
        assert_abs_diff_eq!(big_dense, ref_dense, epsilon = 1e-12);
    }
}