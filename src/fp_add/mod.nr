pub mod constants;

use constants::NUM_REPS;

pub struct FPAddWitness {
    a: Field,
    b: Field,
    c: Field, // c should be approximately a + b
}

pub fn check_fp_add(wit: FPAddWitness) {
    // Verify: a + b - c + 3 is small (fits in 3 bits)
    // This checks that c is approximately a + b with bounded error
    let error = wit.a + wit.b - wit.c + 3;
    error.assert_max_bit_size::<2>();
}

pub fn run_fp_add_experiment(wits: [FPAddWitness; NUM_REPS]) {
    for wit in wits {
        check_fp_add(wit);
    }
}
