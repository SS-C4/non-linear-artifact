// Approximation using Gauss-Legendre quadrature
pub mod gl_quad;
use gl_quad::{constants::{FUNC_TYPE as GL_FUNC_TYPE, NUM_INPUTS as GL_NUM}, GLWitness};

// Lookup based approximation
pub mod lookup;
use lookup::{constants::{FUNC_TYPE as LOOKUP_FUNC_TYPE, NUM_INPUTS as LOOKUP_NUM}, LookupWitness};

// Piecewise linear approximation
pub mod pwl;
use pwl::{constants::NUM_INPUTS as PWL_NUM, PWLWitness};

// Polynomial approximation
pub mod poly_approx;
use poly_approx::{constants::NUM_INPUTS as POLY_NUM, PolyWitness};

// Pade approximation
pub mod pade_approx;
use pade_approx::{constants::NUM_INPUTS as PADE_NUM, PadeWitness};

// Experiment functions
#[allow(dead_code)]
fn run_lookup_experiment(wits: [LookupWitness<LOOKUP_FUNC_TYPE>; LOOKUP_NUM]) {
    for wit in wits {
        lookup::check_lookup(wit);
    }
}

#[allow(dead_code)]
fn run_gl_quad_experiment(wits: [GLWitness<GL_FUNC_TYPE>; GL_NUM]) {
    for wit in wits {
        gl_quad::check_gl_quad(wit);
    }
}

#[allow(dead_code)]
fn run_pwl_experiment(wits: [PWLWitness; PWL_NUM]) {
    for wit in wits {
        pwl::check_pwl_approx(wit);
    }
}

#[allow(dead_code)]
fn run_poly_experiment(wits: [PolyWitness; POLY_NUM]) {
    for wit in wits {
        poly_approx::check_poly_approx(wit);
    }
}

#[allow(dead_code)]
fn run_pade_experiment(wits: [PadeWitness; PADE_NUM]) {
    for wit in wits {
        pade_approx::check_pade_approx(wit);
    }
}

// // Main function for LOOKUP
// fn  main(lookup_wits: [LookupWitness<LOOKUP_FUNC_TYPE>; LOOKUP_NUM]) {
//     run_lookup_experiment(lookup_wits);
// }

// // Main function for GL_QUAD
// fn main(gl_wits: [GLWitness<GL_FUNC_TYPE>; GL_NUM]) {
//     run_gl_quad_experiment(gl_wits);
// }

// // Main function for PWL
fn main(pwl_wits: [PWLWitness; PWL_NUM]) {
    run_pwl_experiment(pwl_wits);
}

// // Main function for POLY
// fn main(poly_wits: [PolyWitness; POLY_NUM]) {
//     run_poly_experiment(poly_wits);
// }

// // Main function for PADE
// fn main(pade_wits: [PadeWitness; PADE_NUM]) {
//     run_pade_experiment(pade_wits);
// }
