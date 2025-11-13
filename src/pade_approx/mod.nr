pub mod constants;

use constants::{LOG_SCALE, SCALE, NUM_COEFFS, DENOM_COEFFS, DEGREE, IS_POWER, IS_KEPLER, IS_BANCOR, TWO_PI, PI};

pub struct KeplerWitness {
    pub P_orbit: Field,
    pub dt: Field,
    pub e: Field,
    pub x: Field,
    pub y: Field,

    pub a: Field,
    pub b: Field,
    pub a_sq: Field,
    pub sqrt_1_m_e2: Field,
    pub M: Field,
    pub E: Field,
    pub sin_E: Field,
    pub cos_E: Field,
    pub cos_E_shifted: Field,
    pub selector: Field,
}

pub struct BancorWitness {
    pub S_supply: Field,
    pub R_reserve: Field,
    pub E_payment: Field,
    pub F_reserve_ratio: Field,
    pub T_tokens: Field,

    pub pow_input: Field,
    pub pow_output: Field,
    pub pow_intermediates: [Field; DEGREE + 1],
}

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

    // For Kepler
    kepler_witness: KeplerWitness,

    // For Bancor
    bancor_witness: BancorWitness,
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
    if IS_KEPLER {
        let kepler_wit = wit.kepler_witness;
        // Check shifted cosine
        assert(kepler_wit.selector * (kepler_wit.selector - 1) * (kepler_wit.selector - 2) == 0);
        let coeff = 2 * kepler_wit.selector * kepler_wit.selector - 4 * kepler_wit.selector + 1;
        let error = kepler_wit.cos_E_shifted * coeff - kepler_wit.cos_E + 1;
        error.assert_max_bit_size::<1>();

        // Check a_sq = a * a
        let error = kepler_wit.a_sq * SCALE - kepler_wit.a * kepler_wit.a + 4*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();

        // Check a^3 = P_orbit^2
        let error = kepler_wit.P_orbit * kepler_wit.P_orbit - kepler_wit.a * kepler_wit.a_sq + 16*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 5>();

        // Check b = a * sqrt(1 - e^2)
        let error = kepler_wit.b * SCALE - kepler_wit.a * kepler_wit.sqrt_1_m_e2 + 4*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();

        // Check M = 2pi * dt / P_orbit
        let error = kepler_wit.M * kepler_wit.P_orbit - TWO_PI * kepler_wit.dt + 16*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 5>();

        // Check M = E - e * sin(E)
        let error = kepler_wit.M * SCALE - kepler_wit.E * SCALE + kepler_wit.e * kepler_wit.sin_E + 16*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 5>();

        // Check sin_E^2 + cos_E^2 = 1
        let error = kepler_wit.sin_E * kepler_wit.sin_E + kepler_wit.cos_E * kepler_wit.cos_E - SCALE * SCALE + 16*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 5>();

        // Check sqrt_1_m_e2 = sqrt(1 - e^2)
        let error = kepler_wit.sqrt_1_m_e2 * kepler_wit.sqrt_1_m_e2 + kepler_wit.e * kepler_wit.e - SCALE * SCALE + 8*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 4>();

        // Check x = a * (cos_E - e)
        let error = kepler_wit.x * SCALE - kepler_wit.a * (kepler_wit.cos_E - kepler_wit.e) + 4*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();

        // Check y = b * sin_E
        let error = kepler_wit.y * SCALE - kepler_wit.b * kepler_wit.sin_E + 4*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();

        // Check powers
        check_powers(kepler_wit.E - kepler_wit.selector * PI, wit.intermediates);

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

        // Check that approximation is close to cos_E_shifted
        let error = kepler_wit.cos_E_shifted * wit.denom_rescaled - num + 4 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();
    }

    else if IS_POWER {
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

    else if IS_BANCOR {
        let bancor_wit = wit.bancor_witness;

        // Check pow_input = (R + E) / R
        let error = bancor_wit.pow_input * bancor_wit.R_reserve - (bancor_wit.E_payment + bancor_wit.R_reserve) * SCALE + 8*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 4>();

        // Check T_tokens = S_supply * (pow_output - 1)
        let error = bancor_wit.T_tokens * SCALE - bancor_wit.S_supply * (bancor_wit.pow_output - SCALE) + 8*SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 4>();

        // Now do the same as the power case: pow_input^F = exp(F * log(pow_input))

        // Check powers for first exponent: log(pow_input)
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

        // Check that approximation is close to pow_input
        let error = bancor_wit.pow_input * wit.denom_rescaled - num + 2 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 2>();

        // Check powers for second exponent: F * log(pow_input)
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

        // Check that approximation is close to pow_output
        let error_2 = bancor_wit.pow_output * wit.denom_rescaled_2 - num_2 + 2 * SCALE;
        error_2.assert_max_bit_size::<LOG_SCALE + 2>();

        // Check that k_log_x = k * log_x (wit.k = F, wit.k_log_x = F * log(pow_input))
        check_quantized_mult(wit.k, wit.log_x, wit.k_log_x);
    }

    else {
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
        let error = wit.y * wit.denom_rescaled - num + 4 * SCALE;
        error.assert_max_bit_size::<LOG_SCALE + 3>();
    }
}