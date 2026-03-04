use nalgebra::{DMatrix, DVector, Dyn, MatrixView, U1};
use crate::transformations::fft::{FFTType, FFTWwrapper};

pub fn even_power_of_imaginary_unit(degree : usize) -> f64 {
    match degree % 4 {
        0 => 1.0,
        1 => 0.0,
        2 => -1.0,
        3 => 0.0,
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Debug)]
pub enum HilbertBasisType {
    Sine,
    Cosine,
}

#[derive(Clone)]
pub(crate) struct HilbertBasisTransformForSeparateDegrees {
    n_t : usize, //number of intervals = number of sample points
    degree : usize,
    DST : FFTWwrapper,
    DCT : FFTWwrapper,
    basis_type : HilbertBasisType,
}

impl HilbertBasisTransformForSeparateDegrees {
    pub fn new(n_t: usize, degree : usize, basis_type : HilbertBasisType) -> Self {
        // Initialize the sine transform for the given number of degrees
        let DST = FFTWwrapper::new(n_t, FFTType::DST4);
        let DCT = FFTWwrapper::new(n_t, FFTType::DCT4);
        HilbertBasisTransformForSeparateDegrees {
            n_t,
            degree,
            DST,
            DCT,
            basis_type,
        }
    }

    pub fn forward_for_degree(&self, degree : usize, u_d : MatrixView<f64, Dyn, U1>) -> DVector<f64> {
        // Perform the forward transform for a specific degree
        let result = if degree%2 == 0 {
            match self.basis_type {
                HilbertBasisType::Sine => {
                    //even case relates to DST up to a factor
                    let u_sine_d = self.DST.forward(u_d.as_slice());
                    let fac = -even_power_of_imaginary_unit(degree+2);
                    fac*DVector::from_column_slice(&u_sine_d)
                },
                HilbertBasisType::Cosine => {
                    //even case relates to DCT up to a factor
                    let u_sine_d = self.DCT.forward(u_d.as_slice());
                    let fac = even_power_of_imaginary_unit(degree);
                    fac*DVector::from_column_slice(&u_sine_d)
                },
            }
        } else {
            match self.basis_type {
                HilbertBasisType::Sine => {
                    //odd case relates to DCT up to a factor
                    let u_cosine_d = self.DCT.forward(u_d.as_slice());
                    let fac = -even_power_of_imaginary_unit(degree+1);
                    fac*DVector::from_column_slice(&u_cosine_d)
                },
                HilbertBasisType::Cosine => {
                    //odd case relates to DST up to a factor
                    let u_cosine_d = self.DST.forward(u_d.as_slice());
                    let fac = even_power_of_imaginary_unit(degree+1);
                    fac*DVector::from_column_slice(&u_cosine_d)
                },
            }
        };
        result
    }

    pub fn transpose_for_degree(&self, degree : usize, u_d : MatrixView<f64, Dyn, U1>) -> DVector<f64> {
        self.forward_for_degree(degree, u_d) //for DST and DCT the transpose is just itself
    }

    pub fn forward(&self, u: &MatrixView<f64, Dyn, Dyn>) -> DMatrix<f64> {
        // assert that the number of columns is degree + 1
        assert_eq!(u.ncols(), self.degree + 1, "HilbertBasisTransformForSeparateDegrees: forward: Number of columns of input matrix does not match degree + 1");
        // assert that the number of rows is n_t
        assert_eq!(u.nrows(), self.n_t, "HilbertBasisTransformForSeparateDegrees: forward: Number of rows of input matrix does not match n_t");
        // Perform the forward sine transform for each degree
        let mut result = DMatrix::zeros(u.nrows(), u.ncols());
        for d in 0..=self.degree {
            let u_d = u.column(d);
            let transformed_d = self.forward_for_degree(d, u_d);
            result.set_column(d, &transformed_d);
        }
        result
    }

    pub fn transpose(&self, u: &MatrixView<f64, Dyn, Dyn>) -> DMatrix<f64> {
        // for DST and DCT the transpose is just itself
        self.forward(u)
    }

}

#[cfg(test)]
mod tests {
    use crate::transformations::degree_dependent_transformations::{even_power_of_imaginary_unit, HilbertBasisType};

    #[test]
    fn test_even_power_of_imaginary_unit() {
        assert_eq!(even_power_of_imaginary_unit(0), 1.0);
        assert_eq!(even_power_of_imaginary_unit(1), 0.0);
        assert_eq!(even_power_of_imaginary_unit(2), -1.0);
        assert_eq!(even_power_of_imaginary_unit(3), 0.0);
        assert_eq!(even_power_of_imaginary_unit(4), 1.0);
        assert_eq!(even_power_of_imaginary_unit(5), 0.0);
        assert_eq!(even_power_of_imaginary_unit(6), -1.0);
        assert_eq!(even_power_of_imaginary_unit(42), -1.0);
        assert_eq!(even_power_of_imaginary_unit(72), 1.0);
    }

    #[test]
    fn test_sine_roundabout(){
        let n_t = 81;
        let degree = 24;
        let transformer = super::HilbertBasisTransformForSeparateDegrees::new(n_t, degree, HilbertBasisType::Sine);
        let mut input = nalgebra::DMatrix::zeros(n_t, degree+1);
        let mut ind = 1;
        for d in 0..=degree {
            for i in 0..n_t {
                input[(i, d)] = ind as f64;
                ind += 1;
            }
        }
        //get input as view
        let input = input.as_view();
        let transformed = transformer.forward(&input);
        let transformed = transformed.as_view();
        let roundtrip = transformer.transpose(&transformed)*(1.0/(2.0*n_t as f64));
        let tol = 1e-9;
        for i in 0..n_t {
            for d in 0..=degree {
                assert!((input[(i, d)] - roundtrip[(i, d)]).abs() < tol, "i={}, d={}, input={}, roundtrip={}", i, d, input[(i, d)], roundtrip[(i, d)]);
            }
        }
    }
}