pub mod constants;

use constants::{LOG_SCALE, SCALE, NUM_COEFFS, DENOM_COEFFS, DEGREE, IS_POWER};

pub struct PadeWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = Pade approximation at x

    // Witness
    intermediates: [Field; DEGREE + 1], // x^1, x^2, ..., x^DEGREE
    denom_rescaled: Field,

    // For power function only
    k: Field,
    log_x: Field,
    k_log_x: Field,
    intermediates_2: [Field; DEGREE + 1],
    denom_rescaled_2: Field,
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
    if !IS_POWER {
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
        let error = wit.y * wit.denom_rescaled - num + 2 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 2>();
    }
    if IS_POWER {
        // Check powers for first exponent
        check_powers(wit.log_x, wit.intermediates);

        // Reconstruct pade approximation numerator for first exponent
        let mut num = 0;
        for i in 0..DEGREE + 1 {
            let coeff = NUM_COEFFS[i];
            let term = wit.intermediates[i] * coeff;
            num += term;
        }

        // Reconstruct pade approximation denominator for first exponent
        let mut denom = 0;
        for i in 0..DEGREE + 1 {
            let coeff = DENOM_COEFFS[i];
            let term = wit.intermediates[i] * coeff;
            denom += term;
        }

        // Check rescaled denominator for first exponent
        check_quantized_mult(denom, 1, wit.denom_rescaled);

        // Check that approximation is close to x
        let error = wit.x * wit.denom_rescaled - num + 2 * SCALE;
        println(error);
        error.assert_max_bit_size::<LOG_SCALE + 2>();

        // Check powers for second exponent
        check_powers(wit.k_log_x, wit.intermediates_2);

        // Reconstruct pade approximation numerator for second exponent
        let mut num_2 = 0;
        for i in 0..DEGREE + 1 {
            let coeff = NUM_COEFFS[i];
            let term = wit.intermediates_2[i] * coeff;
            num_2 += term;
        }

        // Reconstruct pade approximation denominator for second exponent
        let mut denom_2 = 0;
        for i in 0..DEGREE + 1 {
            let coeff = DENOM_COEFFS[i];
            let term = wit.intermediates_2[i] * coeff;
            denom_2 += term;
        }

        // Check rescaled denominator for second exponent
        check_quantized_mult(denom_2, 1, wit.denom_rescaled_2);

        // Check that approximation is close to y
        let error_2 = wit.y * wit.denom_rescaled_2 - num_2 + 2 * SCALE;
        error_2.assert_max_bit_size::<LOG_SCALE + 2>();

        // Check that k_log_x = k * log_x
        check_quantized_mult(wit.k, wit.log_x, wit.k_log_x);
    }
}

