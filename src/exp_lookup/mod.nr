pub mod constants;

use constants::{LOG_BASE, BASE_POWERS, LOG_S, S, MULT_OPS, NUM_TABLES, EXP_TABLES};

pub struct ExpLookupWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output = exp(-x)

    // Witness
    x_decomp: [Field; NUM_TABLES], // Decomposed input value
    lookup_mults: [Field; 2 * NUM_TABLES - 1], // Intermediate products for quantized multiplication
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
    let error = (a_q * b_q - ab_q * S + 100 * S);
    error.assert_max_bit_size::<LOG_S>();
}

/// Checks all the intermediate products
fn check_intermediate_products(lookup_outputs: [Field; NUM_TABLES - 1], 
                               intermediate_mults: [Field; NUM_TABLES - 1],
                               claimed_product: Field) {
    // Check each level of multiplication
    // The first half are products of the lookup outputs
    for i in 0..NUM_TABLES / 2 {
        check_quantized_mult(lookup_outputs[2*i], lookup_outputs[2*i + 1], intermediate_mults[i]);
    }

}

pub fn check_exp_lookup(wit: ExpLookupWitness) {
    // Check validity of the decomposition
    check_decomp(wit.x, wit.x_decomp);

    // Lookup exp values and reconstruct
    let mut product = 1;
    for i in 0..NUM_TABLES {
        // Check that the first NUM_TABLES elements in lookup_mults are the lookup values
        // product *= coeff;
        // TODO: Use check_quantized_mult to check the product
    }
    
    // TODO: Check that product is close to wit.y

}