pub mod constants;

use constants::{LOG_SCALE, SCALE, COEFFS, DEGREE, IS_POWER};

pub struct PolyWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = taylor series approximation of f(x)

    // Witness
    intermediates: [Field; DEGREE + 1], // x^1, x^2, ..., x^DEGREE

    // For power function only
    k: Field,
    log_x: Field,
    k_log_x: Field,
    intermediates_2: [Field; DEGREE + 1],
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
    if !IS_POWER {
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
    else {
        // Check powers for first exponent
        check_powers(wit.log_x, wit.intermediates);

        // Reconstruct polynomial approximation
        let mut log_approx = 0;
        for i in 0..DEGREE + 1 {
            let coeff = COEFFS[i];
            let term = wit.intermediates[i] * coeff;
            log_approx += term;
        }

        // Check that approximation is close to x
        let error = wit.x * SCALE - log_approx + 128 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 8>();

        // Check powers for second exponent
        check_powers(wit.k_log_x, wit.intermediates_2);

        // Reconstruct polynomial approximation for k*log(x)
        let mut k_log_approx = 0;
        for i in 0..DEGREE + 1 {
            let coeff = COEFFS[i];
            let term = wit.intermediates_2[i] * coeff;
            k_log_approx += term;
        }

        // Check that approximation is close to k*log(x)
        let error = wit.y * SCALE - k_log_approx + 128 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 8>();

        // Check k_log_x is k times log_x
        check_quantized_mult(wit.log_x, wit.k, wit.k_log_x);
    }
}

