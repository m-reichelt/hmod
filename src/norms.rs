use std::f64::consts::PI;
use pyo3::prelude::*;
use super::transformations::fft::{FFTType, FFTWwrapper};

#[pyfunction]
pub fn h12_seminorm(f_samples : Vec<f64>) -> PyResult<f64> {
    let n = f_samples.len();
    let h = 1.0 / (n as f64);
    let DST = FFTWwrapper::new(n, FFTType::DST4);
    let f_hat = DST.forward(&f_samples);
    let omegas = (0..n).map(|k| PI*(0.5+k as f64)).collect::<Vec<f64>>();
    let seminorms = f_hat.iter().zip(omegas.iter())
        .map(|(f_k, &omega_k)| {
            omega_k*f_k*f_k
        } );
    // sum the seminorm contributions and scale
    let seminorms_sum = seminorms.sum::<f64>();
    let seminorm_scaled = 0.5*h*h*seminorms_sum;


    Ok(seminorm_scaled.sqrt())
}



#[cfg(test)]
mod tests {
    use super::*;

    fn sine_base_fun(t : f64, k: usize) -> f64 {
        (PI * (k as f64 + 0.5) * t).sin()
    }
    #[test]
    fn test_h12_seminorm() {
        use ndarray::linspace;
        let nt = 1000;
        let t_samples = linspace(0.0, 1.0, nt+1);
        let k = 3;
        let f_samples : Vec<f64> = t_samples.into_iter().map(|t| sine_base_fun(t, k)).collect();
        let seminorm = h12_seminorm(f_samples).unwrap();
        let expected_seminorm = (0.5 * PI * (k as f64 + 0.5)).sqrt();
        let tol = 1e-3;
        assert!((seminorm - expected_seminorm).abs() < tol, "Computed seminorm: {}, Expected seminorm: {}", seminorm, expected_seminorm);
    }
}