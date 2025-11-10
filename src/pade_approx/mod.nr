pub mod constants;

use constants::{LOG_SCALE, SCALE, NUM_COEFFS, DENOM_COEFFS, DEGREE};

pub struct PadeWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = Pade approximation at x

    // Witness
    intermediates: [Field; DEGREE + 1], // x^1, x^2, ..., x^DEGREE
    denom_rescaled: Field,
}

/// Multiplies two quantized values and checks that
/// the quantized product is close to the expected value.
fn check_quantized_mult(a_q: Field, b_q: Field, ab_q: Field) {
    // Multiply and keep the top half
    let error = (a_q * b_q - ab_q * SCALE + 2 * SCALE);
    error.assert_max_bit_size::<LOG_SCALE + 2>();
}

pub fn check_powers(x: Field, intermediates: [Field; DEGREE + 1]) {
    // Check that intermediates are powers of x
    for i in 1..DEGREE + 1 {
        if i == 1 {
            // x^1
            check_quantized_mult(x, SCALE, intermediates[i]);
        } else {
            // x^{i+1} = x^i * x
            check_quantized_mult(intermediates[i - 1], x, intermediates[i]);
        }
    }
}

pub fn check_pade_approx(wit: PadeWitness) {
    // Check powers
    check_powers(wit.x, wit.intermediates);

    // Reconstruct pade approximation numerator
    let mut num = 0;
    for i in 0..DEGREE + 1 {
        let coeff = NUM_COEFFS[i];
        let term = wit.intermediates[i] * coeff;
        num += term;
    }

    // Reconstruct pade approximation denominator
    let mut denom = 0;
    for i in 0..DEGREE + 1 {
        let coeff = DENOM_COEFFS[i];
        let term = wit.intermediates[i] * coeff;
        denom += term;
    }

    // Check rescaled denominator
    check_quantized_mult(denom, 1, wit.denom_rescaled);

    // Check that approximation is close to y
    let error = wit.y * wit.denom_rescaled - num + 32 * SCALE;
    error.assert_max_bit_size::<LOG_SCALE + 6>();
}

