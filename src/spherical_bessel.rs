//! Local spherical Bessel implementation used to avoid a stability issue in
//! `scirs2-special 0.2.0`.
//!
//! The upstream routine is accurate in many regimes, but it can fail
//! catastrophically when the order is much larger than the argument; for
//! example, `j_24(3.2986722862692828)` should be about `4.24e-20`, while
//! scirs2 returned a value of order `1e1` in this dependency version. High
//! polynomial degrees and low Fourier modes can enter exactly this regime.
//! This helper uses a small-argument series, upward recurrence where stable,
//! and Miller/downward recurrence otherwise. It can be revisited once the old
//! brute-force paths are removed or the upstream implementation is fixed.

pub(crate) fn spherical_jn(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }

    let sign = if x < 0.0 && n % 2 == 1 { -1.0 } else { 1.0 };
    let x = x.abs();

    if n == 0 {
        return sign * spherical_j0(x);
    }
    if n == 1 {
        return sign * spherical_j1(x);
    }

    if x > n as f64 {
        sign * spherical_jn_upward(n, x)
    } else if x < 1.0e-2 {
        sign * spherical_jn_series(n, x)
    } else {
        sign * spherical_jn_downward(n, x)
    }
}

fn spherical_j0(x: f64) -> f64 {
    x.sin() / x
}

fn spherical_j1(x: f64) -> f64 {
    x.sin() / (x * x) - x.cos() / x
}

fn spherical_jn_upward(n: usize, x: f64) -> f64 {
    let mut j_prev = spherical_j0(x);
    let mut j_cur = spherical_j1(x);
    for ell in 1..n {
        let j_next = ((2 * ell + 1) as f64 / x) * j_cur - j_prev;
        j_prev = j_cur;
        j_cur = j_next;
    }
    j_cur
}

fn spherical_jn_downward(n: usize, x: f64) -> f64 {
    let l_max = n + 50 + x.ceil() as usize;
    let mut values = vec![0.0; l_max + 2];
    values[l_max] = 1.0;

    for ell in (1..=l_max).rev() {
        values[ell - 1] = ((2 * ell + 1) as f64 / x) * values[ell] - values[ell + 1];
        if values[ell - 1].abs() > 1.0e200 {
            for value in values.iter_mut().take(l_max + 2).skip(ell - 1) {
                *value *= 1.0e-200;
            }
        }
    }

    let j0 = spherical_j0(x);
    if j0.abs() > 1.0e-14 && values[0].abs() > 0.0 {
        return values[n] * j0 / values[0];
    }

    values[n] * spherical_j1(x) / values[1]
}

fn spherical_jn_series(n: usize, x: f64) -> f64 {
    let mut prefactor = 1.0;
    for k in 0..n {
        prefactor *= x / (2 * k + 3) as f64;
    }

    let x2 = x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 0..100 {
        let denominator = 2.0 * (k + 1) as f64 * (2 * k + 2 * n + 3) as f64;
        term *= -x2 / denominator;
        sum += term;
        if term.abs() < 1.0e-16 * sum.abs().max(1.0) {
            break;
        }
    }

    prefactor * sum
}

#[cfg(test)]
mod tests {
    use super::spherical_jn;

    #[test]
    fn test_spherical_jn_large_arguments() {
        let j0_expected = |t: f64| t.sin() / t;
        let j1_expected = |t: f64| t.sin() / (t * t) - t.cos() / t;
        let j2_expected =
            |t: f64| ((3.0 / (t * t) - 1.0) * (t.sin() / t)) - (3.0 * t.cos() / (t * t));

        for i in 1..200 {
            let t = 20.0 + 0.5 * i as f64;
            assert!((spherical_jn(0, t) - j0_expected(t)).abs() < 1e-14);
            assert!((spherical_jn(1, t) - j1_expected(t)).abs() < 1e-14);
            assert!((spherical_jn(2, t) - j2_expected(t)).abs() < 1e-14);
        }
    }

    #[test]
    fn test_spherical_jn_high_degree_reference_values() {
        assert!((spherical_jn(5, 0.1) - 9.61631023291645e-10).abs() < 1e-22);
        assert!((spherical_jn(10, 1.0) - 7.116552640047341e-11).abs() < 1e-23);
        assert!((spherical_jn(24, 3.2986722862692828) - 4.236791202132689e-20).abs() < 1e-30);
        assert!((spherical_jn(24, 21.83406894244906) - 0.01267410900915432).abs() < 1e-14);
        assert!((spherical_jn(24, 31.258846903218444) + 0.023428643566426106).abs() < 1e-14);
    }
}
