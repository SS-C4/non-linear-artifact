use super::constants::{LOG_S, S, S_sq, N_POINTS, GL_ROOTS, GL_ROOTS_tan, GL_WEIGHTS};

// Common Trait
pub trait GLFunction {
    // For all functions
    fn check_y_bound(self, inp: Input);
    fn check_ratio(self, inp: Input);
    fn check_inverse(self, inp: Input);
    fn check_sum(self, inp: Input);

    // For tan(x)
    fn check_mult(self, inp: Input);
}

// Common Statement and witness
pub struct GLWitness <T> {
    // Statement
    pub inp_struct: Input,

    // Witness
    pub wit_struct: T,
}

pub struct Input {
    pub x: Field, // Input
    pub y: Field, // Output
    pub k: Field, // For power function
}

// All possible function types
// y = e^x
pub struct ExponentialWitness {
    pub ratio_term: Field, 
    pub denom_inverses: [Field; N_POINTS],
}

// y = e^-x
pub struct InverseExponentialWitness {
    pub ratio_term: Field, 
    pub denom_inverses: [Field; N_POINTS],
}

// y = sigmoid(x) = 1 / (1 + e^-x)
pub struct SigmoidWitness {
    pub ratio_term: Field, 
    pub denom_inverses: [Field; N_POINTS],
}

// y = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
pub struct TanhWitness {
    pub ratio_term: Field, 
    pub denom_inverses: [Field; N_POINTS],
}

// y = tan(x)
pub struct TangentWitness {
    pub ratio_term: Field, 
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = x^k
pub struct PowerWitness {
    pub ratio_term_1: Field, 
    pub ratio_term_2: Field,
    pub denom_inverses_1: [Field; N_POINTS],
    pub denom_inverses_2: [Field; N_POINTS],
}

// Implementations
impl GLFunction for ExponentialWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for e^x
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * (y - 1) ~= (y + 1)
        let error = (S + inp.y) * S - self.ratio_term * (inp.y - S) + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, _inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] + self.ratio_term;
            let error = S_sq - term * self.denom_inverses[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 20>();
    }

    fn check_mult(self, _inp: Input) {
        // No multiplicative checks needed for exp(x)
    }
}

impl GLFunction for InverseExponentialWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * (1 - y) ~= (1 + y)
        let error = (S + inp.y) * S - self.ratio_term * (S - inp.y) + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, _inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] + self.ratio_term;
            let error = S_sq - term * self.denom_inverses[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 13>();
    }

    fn check_mult(self, _inp: Input) {
        // No multiplicative checks needed for e^-x
    }
}

impl GLFunction for SigmoidWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * (2y - 1) ~= 1
        let error = S_sq - self.ratio_term * (2*inp.y - S) + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, _inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] + self.ratio_term;
            let error = S_sq - term * self.denom_inverses[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 13>();
    }

    fn check_mult(self, _inp: Input) {
        // No multiplicative checks needed for sigmoid
    }
}

impl GLFunction for TanhWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * y ~= 2
        let error = 2*S_sq - self.ratio_term * inp.y + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, _inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = 2 * GL_ROOTS[i] + self.ratio_term;
            let error = S_sq - term * self.denom_inverses[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 13>();
    }

    fn check_mult(self, _inp: Input) {
        // No multiplicative checks needed for tanh
    }
}

impl GLFunction for TangentWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * y ~= 2
        let error = 2*S_sq - self.ratio_term * inp.y + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, _inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = self.mult_terms[i] + self.ratio_term;
            let error = S_sq - term * self.denom_inverses[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 13>();
    }

    fn check_mult(self, inp: Input) {
        // Check multiplicative terms
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] - GL_ROOTS_tan[i] * inp.y + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }
}

impl GLFunction for PowerWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for power function
    }

    fn check_ratio(self, inp: Input) {
        // Check that ratio * (1 - y) ~= (1 + y)
        let error = self.ratio_term_1 * (S - inp.y) - (S + inp.y) + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();

        // Check that ratio * (1 - x) ~= (1 + x)
        let error = self.ratio_term_2 * (S - inp.x) - (S + inp.x) + 10000*S;
        error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses for first ratio
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] + self.ratio_term_1;
            let error = S_sq - term * self.denom_inverses_1[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
        // Check inverses for second ratio
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] + self.ratio_term_2;
            let error = inp.k * S - term * self.denom_inverses_2[i] + 10000*S;
            error.assert_max_bit_size::<LOG_S + 15>();
        }
    }

    fn check_sum(self, _inp: Input) {
        // Check the sum
        let lhs = 0;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * (self.denom_inverses_1[i] - self.denom_inverses_2[i]);
        }
        let error = lhs - rhs + 10*S;
        error.assert_max_bit_size::<LOG_S + 13>();
    }

    fn check_mult(self, _inp: Input) {
        // No multiplicative checks needed for power
    }
}