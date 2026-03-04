#[cfg(test)]
mod tests {
    use super::super::{FFTWwrapper, FFTType};
    #[test]
    fn roundtrip() {
        let n = 8;
        let t = FFTWwrapper::new(n, FFTType::DST4);
        let x: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5).sin()).collect();
        let y = t.forward(&x);
        let x_rt = t.transpose(&y);
        // for the transpose to be the inverse, we need to normalize by 2n
        let x_rt: Vec<f64> = x_rt.iter().map(|v| v / (2.0 * n as f64)).collect();
        // allow a tiny numerical tolerance
        for (a, b) in x.iter().zip(x_rt.iter()) {
            assert!((a - b).abs() < 1e-9, "a={a}, b={b}");
        }
    }

    fn generate_sample(n: usize, mode : usize) -> Vec<f64> {
        use std::f64::consts::PI;
        let mode = mode as f64;
        let n_i = n;
        let n = n as f64;
        let basis_function = |i: f64| (PI*(i+0.5)*(mode+0.5)/n).sin();
        (0..n_i).map(|i| basis_function(i as f64)).collect()
    }

    #[test]
    fn test_known_values() {
        let n = 32;
        let t = FFTWwrapper::new(n, FFTType::DST4);
        let mode = 1;
        let x = generate_sample(n, mode);
        let y = t.forward(&x);
        let mut expected_y = vec![0.; n];
        expected_y[mode] = n as f64;
        for (a, b) in y.iter().zip(expected_y.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a}, b={b}");
        }
    }
}
