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

fn main(
    _lookup_wits: [LookupWitness<LOOKUP_FUNC_TYPE>; LOOKUP_NUM],
    // _gl_wits: [GLWitness<GL_FUNC_TYPE>; GL_NUM],
    // _pwl_wits: [PWLWitness; PWL_NUM],
    // _poly_wits: [PolyWitness; POLY_NUM],
    // _pade_wits: [PadeWitness; PADE_NUM],
) {
    for lookup_wit in _lookup_wits {
        lookup::check_lookup(lookup_wit);
    }
    // for cosine_wit in _cosine_wits {
    //     exp_lookup::check_cosine_lookup(cosine_wit);
    // }
    // for gl_wit in _gl_wits {
    //     gl_quad::check_gl_quad(gl_wit);
    // }
    // for pwl_wit in _pwl_wits {
    //     pwl::check_pwl_approx(pwl_wit);
    // }
    // for poly_wit in _poly_wits {
    //     poly_approx::check_poly_approx(poly_wit);
    // }
    // for pade_wit in _pade_wits {
    //     pade_approx::check_pade_approx(pade_wit);
    // }
    // for softmax_wit in _softmax_wits {
    //     exp_lookup::check_softmax_lookup(softmax_wit);
    // }
}
