pub mod constants;

use constants::{LOG_S, S, S_sq, N_POINTS, GL_ROOTS, GL_WEIGHTS};

// Statement and witness
pub struct GLWitness {
    // Statement
    pub x: Field, // Input value
    pub y: Field, // Output = exp(-x)

    // Witness
    pub ratio: Field, // Ratio = (1 + y) / (1 - y)
    pub gl_inverses: [Field; N_POINTS], // Inverses of root_i + (1 + y) / (1 - y)
}

fn check_ratio(y: Field, ratio: Field) {
    // Check that ratio * (1 - y) ~= (1 + y)
    let error = (S + y) * S - ratio * (S - y) + 100*S;
    error.assert_max_bit_size::<LOG_S + 10>();
}

fn check_inverse(x: Field, x_inv: Field) {
    // Check that x * x_inv ~= 1
    let error = S_sq - x * x_inv + 100*S;
    error.assert_max_bit_size::<LOG_S + 10>();
}

pub fn check_gl_quad(wit: GLWitness) {
    // Check that y is at most 1
    wit.y.assert_max_bit_size::<LOG_S>();

    // Check ratio
    check_ratio(wit.y, wit.ratio);

    // Check inverses
    for i in 0..N_POINTS {
        let term = GL_ROOTS[i] + wit.ratio;
        check_inverse(term, wit.gl_inverses[i]);
    }

    // Check the sum
    let lhs = wit.x * S;
    let mut rhs = 0;
    for i in 0..N_POINTS {
        rhs += GL_WEIGHTS[i] * wit.gl_inverses[i];
    }
    let error = lhs - rhs + 1000*S;
    error.assert_max_bit_size::<LOG_S + 50>();
}