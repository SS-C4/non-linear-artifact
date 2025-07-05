pub mod constants;

use constants::{LOG_BASE, BASE_POWERS, LOG_S, S, BASE, NUM_TABLES, EXP_TABLES};

pub struct ExpLookupWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output = exp(-x)

    // Witness
    x_decomp: [Field; NUM_TABLES], // Decomposed input value
    // TODO: Add the mult results that are checked by check_quantized_mult
}

fn check_decomp(x: Field, decomp: [Field; NUM_TABLES]) {
    // Check that the decomposition reconstructs the original value
    let mut reconstructed = 0;
    for i in 0..NUM_TABLES {
        reconstructed += decomp[i] * BASE_POWERS[i];
    }
    assert(reconstructed == x);

    // Check that all the decomposed values < BASE
    for i in 0..NUM_TABLES {
        decomp[i].assert_max_bit_size::<LOG_BASE>();
    }
}

/// Multiplies two quantized values and checks that
/// the quantized product is close to the expected value.
fn check_quantized_mult(a_q: Field, b_q: Field, ab_q: Field) {
    // Multiply and keep the top half
    // TODO: Add some tolerance: 100 * S or something similar
    let error = (a_q * b_q - ab_q * S);
    error.assert_max_bit_size::<LOG_S>();
}

pub fn check_exp_lookup(wit: ExpLookupWitness) {
    // Check validity of the decomposition
    check_decomp(wit.x, wit.x_decomp);

    // Lookup exp values and reconstruct
    let mut product = 1;
    for i in 0..NUM_TABLES {
        let coeff = EXP_TABLES[i][wit.x_decomp[i] as u32];
        // product *= coeff;
        // TODO: Use check_quantized_mult to check the product
    }
    
    // TODO: Check that product is close to wit.y

}