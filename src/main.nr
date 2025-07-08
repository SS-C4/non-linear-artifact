// Approximation for exp(-x) using Newton-Cotes integration
pub mod nc_int;
use nc_int::{NC_witness, check_nc_int};

// Lookup based approach for exp(-x)
pub mod exp_lookup;
use exp_lookup::{ExpLookupWitness, check_exp_lookup};

fn main(nc_wit: NC_witness, exp_wit: ExpLookupWitness) {
    check_nc_int(nc_wit);
    check_exp_lookup(exp_wit);
}