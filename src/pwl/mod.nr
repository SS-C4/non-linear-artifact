pub mod constants;

use constants::{LINEAR_COEFFS, LOG_SCALE, SCALE, START_POINTS, PIECE_SIZE};

pub struct PWLWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = piecewise linear approximation of f(x)

    // Witness
    segment_index: Field, // Index of the piecewise segment
}

pub fn check_pwl_approx(wit: PWLWitness) {
    // Check that x is in the right segment
    let segment_start = START_POINTS[wit.segment_index as u32];    
    let segment_end = START_POINTS[wit.segment_index as u32] + PIECE_SIZE;

    assert(segment_start.lt(wit.x));
    assert(wit.x.lt(segment_end));

    // Check that y is equal to the linear approximation
    let (a, b) = LINEAR_COEFFS[wit.segment_index as u32];
    let approx_y = (a * wit.x + b * SCALE);
    let error = wit.y * SCALE - approx_y;
    println(error);
    println(SCALE * 2_147_483_648 * 2_147_483_648 * 2_147_483_648);
    error.assert_max_bit_size::<(LOG_SCALE + 60)>();
}