// Approximation for exp(-x) using Newton-Cotes integration
pub mod nc_int;
use nc_int::{NC_witness, constants::NUM_INPUTS as NC_NUM};

// Approximation using Gauss-Legendre quadrature
pub mod gl_quad;
use gl_quad::{GLWitness, constants::{NUM_INPUTS as GL_NUM, FUNC_TYPE}};

// Lookup based approach for exp(-x)
pub mod exp_lookup;
use exp_lookup::{InvExpLookupWitness, SoftMaxLookupWitness, constants::NUM_INPUTS as LOOKUP_NUM};

fn main(
    _nc_wits: [NC_witness; NC_NUM], 
    _exp_wits: [InvExpLookupWitness; LOOKUP_NUM], 
    _gl_wits: [GLWitness<FUNC_TYPE>; GL_NUM], 
    // _softmax_wits: [SoftMaxLookupWitness; LOOKUP_NUM]
) {
    // for nc_wit in _nc_wits {
    //     check_nc_int(nc_wit);
    // }
    for exp_wit in _exp_wits {
        exp_lookup::check_inv_exp_lookup(exp_wit);
    }
    // for softmax_wit in _softmax_wits {
    //     exp_lookup::check_softmax_lookup(softmax_wit);
    // }
    // for gl_wit in _gl_wits {
    //     gl_quad::check_gl_quad(gl_wit);
    // }
}