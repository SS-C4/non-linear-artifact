// Approximation for exp(-x) using Newton-Cotes integration
pub mod nc_int;
use nc_int::{NC_witness, check_nc_int, constants::NUM_INPUTS as NC_NUM};

// Approximation using Gauss-Legendre quadrature
pub mod gl_quad;
use gl_quad::{GLWitness, check_gl_quad, constants::NUM_INPUTS as GL_NUM};

// Lookup based approach for exp(-x)
pub mod exp_lookup;
use exp_lookup::{ExpLookupWitness, check_exp_lookup, constants::NUM_INPUTS as EXP_NUM};

fn main(nc_wits: [NC_witness; NC_NUM], exp_wits: [ExpLookupWitness; EXP_NUM], gl_wits: [GLWitness; GL_NUM]) {
    for nc_wit in nc_wits {
        check_nc_int(nc_wit);
    }
    for exp_wit in exp_wits {
        check_exp_lookup(exp_wit);
    }
    for gl_wit in gl_wits {
        check_gl_quad(gl_wit);
    }
}