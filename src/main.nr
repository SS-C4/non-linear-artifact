// Approximation using Gauss-Legendre quadrature
pub mod gl_quad;
use gl_quad::{constants::{FUNC_TYPE, NUM_INPUTS as GL_NUM}, GLWitness};

// Lookup based approach for exp(-x)
pub mod exp_lookup;
use exp_lookup::{constants::NUM_INPUTS as LOOKUP_NUM, InvExpLookupWitness, SoftMaxLookupWitness, CosineWitness};

// Piecewise linear approximation
pub mod pwl;
use pwl::{constants::NUM_INPUTS as PWL_NUM, PWLWitness};

// Polynomial approximation
pub mod poly_approx;
use poly_approx::{constants::NUM_INPUTS as POLY_NUM, PolyWitness};

fn main(
    // _exp_wits: [InvExpLookupWitness; LOOKUP_NUM],
    // _cosine_wits: [CosineWitness; LOOKUP_NUM],
    _gl_wits: [GLWitness<FUNC_TYPE>; GL_NUM],
    // _pwl_wits: [PWLWitness; PWL_NUM],
    // _poly_wits: [PolyWitness; POLY_NUM],
    // _softmax_wits: [SoftMaxLookupWitness; LOOKUP_NUM]
) {
    // for exp_wit in _exp_wits {
    //     exp_lookup::check_inv_exp_lookup(exp_wit);
    // }
    // for cosine_wit in _cosine_wits {
    //     exp_lookup::check_cosine_lookup(cosine_wit);
    // }
    for gl_wit in _gl_wits {
        gl_quad::check_gl_quad(gl_wit);
    }
    // for pwl_wit in _pwl_wits {
    //     pwl::check_pwl_approx(pwl_wit);
    // }
    // for poly_wit in _poly_wits {
    //     poly_approx::check_poly_approx(poly_wit);
    // }
    // for softmax_wit in _softmax_wits {
    //     exp_lookup::check_softmax_lookup(softmax_wit);
    // }
}
