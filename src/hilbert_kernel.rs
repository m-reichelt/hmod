use scirs2_special::spherical_jn;
use std::f64::consts::PI;
use rayon::prelude::*;

fn first_branch_sum_brute_force(
    diagonal_index: usize,
    nt: usize,
    pol_deg_trial: u32,
    pol_deg_test: u32,
) -> f64 {
    debug_assert!(nt > diagonal_index);
    let nmodes = (1e5/2.) as usize;
    let alpha_k = |k| PI * ((2 * k + 1) as f64) / ((4 * nt) as f64);
    let alpha_q = |q| alpha_k(2 * q * nt + diagonal_index);
    let summand = |q| {
        spherical_jn(pol_deg_test as i32, alpha_q(q))
            * spherical_jn(pol_deg_trial as i32, alpha_q(q))
    };
    let res = (0..nmodes).rev().map(summand).sum::<f64>();
    res / (nt.pow(2) as f64)
}

fn second_branch_sum_brute_force(
    diagonal_index: usize,
    nt: usize,
    pol_deg_trial: u32,
    pol_deg_test: u32,
) -> f64 {
    debug_assert!(nt > diagonal_index);
    let nmodes = (1e5/2.) as usize;
    let alpha_k = |k| PI * ((2 * k + 1) as f64) / ((4 * nt) as f64);
    let alpha_q = |q| alpha_k(2 * nt * (q + 1) - 1 - diagonal_index);
    let summand = |q| {
        spherical_jn(pol_deg_test as i32, alpha_q(q))
            * spherical_jn(pol_deg_trial as i32, alpha_q(q))
    };
    let res = (0..nmodes).rev().map(summand).sum::<f64>();
    res / (nt.pow(2) as f64)
}

pub fn get_kernel_matrix_for_degrees(nt: usize, pol_deg_trial: u32, pol_deg_test: u32) -> Vec<f64> {
    debug_assert!(nt > 0);
    let fac = if pol_deg_trial % 2 != pol_deg_test % 2 {
        1.
    } else {
        -1.
    };
    let entry_i = |i| {
        first_branch_sum_brute_force(i, nt, pol_deg_trial, pol_deg_test)
            + fac * second_branch_sum_brute_force(i, nt, pol_deg_trial, pol_deg_test)
    };
    (0..nt).into_par_iter().with_min_len(8).map(entry_i).collect()
}
