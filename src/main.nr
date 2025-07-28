// Approximation for exp(-x) using Newton-Cotes integration
pub mod nc_int;
use nc_int::{NC_witness, check_nc_int};

// Approximation using Gauss-Legendre quadrature
pub mod gl_quad;
use gl_quad::{GLWitness, check_gl_quad};

// Lookup based approach for exp(-x)
pub mod exp_lookup;
use exp_lookup::{ExpLookupWitness, check_exp_lookup};

fn main(nc_wit: NC_witness, exp_wit: ExpLookupWitness, gl_wit: GLWitness) {
    check_nc_int(nc_wit);
    check_exp_lookup(exp_wit);
    check_gl_quad(gl_wit);
}