pub mod constants;

use constants::{NUM_REPS, SCALE, LOG_SCALE};

pub struct FPMulWitness {
    a: Field,
    b: Field,
    c: Field, // c should be approximately a * b (in fixed-point: a * b = c * S)
}

pub fn check_fp_mul(wit: FPMulWitness) {
    // Verify: a * b - c * S + 4 * S is small (fits in LOG_SCALE + 3 bits)
    // This checks that c is approximately a * b with bounded error
    let error = wit.a * wit.b - wit.c * SCALE + 4 * SCALE;
    error.assert_max_bit_size::<(LOG_SCALE + 3)>();
}

pub fn run_fp_mul_experiment(wits: [FPMulWitness; NUM_REPS]) {
    for wit in wits {
        check_fp_mul(wit);
    }
}
