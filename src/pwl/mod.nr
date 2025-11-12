pub mod constants;

use constants::{LINEAR_COEFFS, LOG_SCALE, SCALE, START_POINTS, PIECE_SIZE, IS_POWER};

pub struct PWLWitness {
    // Statement
    x: Field, // Input value
    y: Field, // Output value = piecewise linear approximation of f(x)

    // Witness
    segment_index: Field, // Index of the piecewise segment

    // For power function only
    k: Field,
    log_x: Field,
    k_log_x: Field,
    segment_index_2: Field,
}

pub fn check_pwl_approx(wit: PWLWitness) {
    if !IS_POWER {    
        // Check that x is in the right segment
        let segment_start = START_POINTS[wit.segment_index as u32];    
        let segment_end = START_POINTS[wit.segment_index as u32] + PIECE_SIZE;

        assert(segment_start.lt(wit.x+1));
        assert(wit.x.lt(segment_end+1));

        // Check that y is equal to the linear approximation
        let (a, b) = LINEAR_COEFFS[wit.segment_index as u32];
        let approx_y = (a * wit.x + b * SCALE);
        let error = wit.y * SCALE - approx_y + 16 * SCALE;
        error.assert_max_bit_size::<(LOG_SCALE + 5)>();
    }
    else {
        // Check that log_x is in the right segment
        let segment_start = START_POINTS[wit.segment_index as u32];    
        let segment_end = START_POINTS[wit.segment_index as u32] + PIECE_SIZE;
        println(wit.log_x);
        println(segment_start);
        println(segment_end);

        assert(segment_start.lt(wit.log_x+1));
        assert(wit.log_x.lt(segment_end+1));

        // Check that x is equal to exp(log_x)
        let (a, b) = LINEAR_COEFFS[wit.segment_index as u32];
        let approx_x = (a * wit.log_x + b * SCALE);
        let error_x = wit.x * SCALE - approx_x + 16 * SCALE;
        error_x.assert_max_bit_size::<(LOG_SCALE + 5)>();

        // Check that k_log_x is in the right segment
        let segment_start_2 = START_POINTS[wit.segment_index_2 as u32];    
        let segment_end_2 = START_POINTS[wit.segment_index_2 as u32] + PIECE_SIZE;

        assert(segment_start_2.lt(wit.k_log_x+1));
        assert(wit.k_log_x.lt(segment_end_2+1));

        // Check that y is equal to exp(k_log_x)
        let (a_2, b_2) = LINEAR_COEFFS[wit.segment_index_2 as u32];
        let approx_y = (a_2 * wit.k_log_x + b_2 * SCALE);
        let error_y = wit.y * SCALE - approx_y + 16 * SCALE;
        error_y.assert_max_bit_size::<(LOG_SCALE + 5)>();
    }
}