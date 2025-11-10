pub mod constants;

use constants::{LOG_SCALE, SCALE, COEFFS, DEGREE};

pub struct PolyWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = taylor series approximation of f(x)

    // Witness
    intermediates: [Field; DEGREE + 1], // x^1, x^2, ..., x^DEGREE
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

pub fn check_poly_approx(wit: PolyWitness) {
    // Check powers
    check_powers(wit.x, wit.intermediates);

    // Reconstruct polynomial approximation
    let mut approx = 0;
    for i in 0..DEGREE + 1 {
        let coeff = COEFFS[i];
        let term = wit.intermediates[i] * coeff;
        approx += term;
    }

    // Check that approximation is close to y
    let error = wit.y * SCALE - approx + 128 * SCALE;
    error.assert_max_bit_size::<LOG_SCALE + 8>();
}

