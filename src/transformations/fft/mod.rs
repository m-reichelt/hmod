
use rustdct::DctPlanner;


#[derive(Clone, Debug, Copy)]
pub enum FFTType {
    DCT4,
    DST4,
    DST2,
    DST3,
}

#[derive(Debug, Clone, Copy)]
pub struct FFTWwrapper {
    n : usize, //number of sample points
    my_type : FFTType,
}


impl FFTWwrapper {
    pub fn new(n: usize, fftw_type : FFTType) -> Self{
        Self {
            n,
            my_type: fftw_type,
        }
    }

    /// y = DST-IV(x)
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n);
        let mut planner = DctPlanner::new();
        let mut buffer = x.to_vec();
        let fft_result = match self.my_type {
            FFTType::DCT4 => {
                let mut transform = planner.plan_dct4(self.n);
                transform.process_dct4(&mut buffer);
                //multiply by 2 to match FFTW definition
                buffer.iter_mut().for_each(|v| *v *= 2.0);
                buffer
            },
            FFTType::DST4 => {
                let mut transform = planner.plan_dst4(self.n);
                transform.process_dst4(&mut buffer);
                //multiply by 2 to match FFTW definition
                buffer.iter_mut().for_each(|v| *v *= 2.0);
                buffer
            }
            FFTType::DST2 => {
                let mut transform = planner.plan_dst2(self.n);
                transform.process_dst2(&mut buffer);
                //multiply by 2 to match FFTW definition
                buffer.iter_mut().for_each(|v| *v *= 2.0);
                buffer
            }
            FFTType::DST3 => {
                let mut transform = planner.plan_dst3(self.n);
                transform.process_dst3(&mut buffer);
                //multiply by 2 to match FFTW definition
                buffer.iter_mut().for_each(|v| *v *= 2.0);
                buffer
            }
        };
        fft_result
    }

    /// x = DST-IV^{-1}(y), is transpose operation
    pub fn transpose(&self, y: &[f64]) -> Vec<f64> {
        match self.my_type {
            FFTType::DST4 => self.forward(y), //DST-IV is its own inverse (up to factor 2N)
            FFTType::DCT4 => self.forward(y), //DCT-IV is its own inverse (up to factor 2N)
            FFTType::DST2 => panic!("Transpose not implemented for DST2. If you need it you likely made an error."),
            FFTType::DST3 => panic!("Transpose not implemented for DST3. If you need it you likely made an error."),
        }
    }
}


//modules for testing specific transformations
mod dst_tests;
mod dct_tests;