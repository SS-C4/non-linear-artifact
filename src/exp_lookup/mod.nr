pub mod constants;

use constants::{LOG_BASE, BASE_POWERS, LOG_S, S, MULT_OPS, NUM_TABLES, NUM_SOFTMAX, EXP_TABLES};

pub struct InvExpLookupWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output = exp(-x)

    // Witness
    x_decomp: [Field; NUM_TABLES], // Decomposed input value
    lookup_mults: [Field; 2 * NUM_TABLES - 1], // Intermediate products for quantized multiplication
}

pub struct SoftMaxLookupWitness {
    // Statement
    vec_x: [Field; NUM_SOFTMAX], // Input vector
    vec_y: [Field; NUM_SOFTMAX], // Output vector = softmax(vec_x)

    // Witnesses
    exp_wits: [InvExpLookupWitness; NUM_SOFTMAX], // Exponential lookup witnesses
    shift: Field,        // Log of denominator
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
    let error = (a_q * b_q - ab_q * S + 16 * S);
    error.assert_max_bit_size::<LOG_S + 5>();
}

/// Checks all the intermediate products
fn check_products(lookup_mults: [Field; 2 * NUM_TABLES - 1]) {
    // Check each multiplication in mult_ops
    for i in 0..MULT_OPS.len() {
        let (a_idx, b_idx, ab_idx) = MULT_OPS[i];
        check_quantized_mult(
            lookup_mults[a_idx as u32],
            lookup_mults[b_idx as u32],
            lookup_mults[ab_idx as u32],
        );
    }
}

pub fn check_inv_exp_lookup(wit: InvExpLookupWitness) {
    // Check validity of the decomposition
    check_decomp(wit.x, wit.x_decomp);

    // Lookup check
    for i in 0..NUM_TABLES {
        // Check that the first NUM_TABLES elements in lookup_mults are the lookup values
        assert(wit.lookup_mults[i] == EXP_TABLES[i][wit.x_decomp[i] as u32]);
    }

    // Multiplication checks
    check_products(wit.lookup_mults);

    // Output range check
    let error = (wit.y - wit.lookup_mults[2 * NUM_TABLES - 2] + 16);
    error.assert_max_bit_size::<5>();
}

pub fn check_softmax_lookup(wit: SoftMaxLookupWitness) {
    // Check that exp_wits[i].x == shift - vec_x[i]
    for i in 0..NUM_SOFTMAX {
        assert(wit.exp_wits[i].x == wit.shift - wit.vec_x[i]);
    }

    // Check each exp lookup
    for i in 0..NUM_SOFTMAX {
        check_inv_exp_lookup(wit.exp_wits[i]);
    }

    // Check sum ~= 1
    let mut sum_exp = 0;
    for i in 0..NUM_SOFTMAX {
        sum_exp += wit.exp_wits[i].y;
    }
    let error = (sum_exp - S + 8);
    error.assert_max_bit_size::<4>();

    // Check vec_y[i] ~= exp(vec_x[i])
    for i in 0..NUM_SOFTMAX {
        let error = (wit.vec_y[i] - wit.exp_wits[i].y);
        error.assert_max_bit_size::<4>();
    }
}