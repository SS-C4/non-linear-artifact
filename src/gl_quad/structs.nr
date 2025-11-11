use super::constants::{LOG_S, S, S_sq, N_POINTS, N_SOFTMAX, GL_ROOTS, GL_ROOTS_tan, GL_WEIGHTS};

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
    pub x: Field, // Scalar Input
    pub y: Field, // Scalar Output

    pub vec_x: [Field; N_SOFTMAX], // For softmax
    pub vec_y: [Field; N_SOFTMAX], // For softmax

    pub k: Field, // For power function
}

// All possible function types
// y = e^x
pub struct ExponentialWitness {
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = e^-x
pub struct InverseExponentialWitness {
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = sigmoid(x) = 1 / (1 + e^-x)
pub struct SigmoidWitness {
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = tanh(x) = (e^x - e^-x) / (e^x + e^-x)
pub struct TanhWitness {
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = tan(x)
pub struct TangentWitness {
    pub y_sq: Field, 
    pub denom_inverses: [Field; N_POINTS],
    pub mult_terms: [Field; N_POINTS],
}

// y = cos(x)
pub struct CosineWitness {
    pub tan_x: Field, // tan(x)
    pub tan_witness: TangentWitness,
    pub sqrt_1_p_tan2: Field, // sqrt(1 + tan^2(x))
}

// y = x^k
pub struct PowerWitness {
    pub denom_inverses_1: [Field; N_POINTS],
    pub denom_inverses_2: [Field; N_POINTS],
    pub mult_terms_1: [Field; N_POINTS],
    pub mult_terms_2: [Field; N_POINTS],
}

// y = softmax(x)
pub struct SoftmaxWitness {
    pub exp_witnesses: [ExponentialWitness; N_SOFTMAX],
    pub exp_outputs: [Field; N_SOFTMAX],
    pub denom_inverse: Field,
}
 
// Implementations
impl GLFunction for ExponentialWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for e^x
    }

    fn check_ratio(self, _inp: Input) {
        // // Check that ratio * (y - 1) ~= (y + 1)
        // let error = (S + inp.y) * S - self.ratio_term * (inp.y - S) + 8192*S;
        // error.assert_max_bit_size::<LOG_S + 14>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = self.mult_terms[i] - GL_ROOTS[i] + inp.y + S;
            let error = (inp.y - S) * S - term * self.denom_inverses[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }

    fn check_mult(self, inp: Input) {
        // Check y * t
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - inp.y * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for InverseExponentialWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, _inp: Input) {
        // // Check that ratio * (1 - y) ~= (1 + y)
        // let error = (S + inp.y) * S - self.ratio_term * (S - inp.y) + 10000*S;
        // error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] - self.mult_terms[i] + S + inp.y;
            let error = (S - inp.y) * S - term * self.denom_inverses[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }

    fn check_mult(self, _inp: Input) {
        // Check y * t
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - _inp.y * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for SigmoidWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, _inp: Input) {
        // // Check that ratio * (2y - 1) ~= 1
        // let error = S_sq - self.ratio_term * (2*inp.y - S) + 10000*S;
        // error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = 2 * self.mult_terms[i] - GL_ROOTS[i] + S;
            let error = (2*inp.y - S) * S - term * self.denom_inverses[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }

    fn check_mult(self, inp: Input) {
        // Check y * t
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - inp.y * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for TanhWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, _inp: Input) {
        // // Check that ratio * y ~= 2
        // let error = 2*S_sq - self.ratio_term * inp.y + 10000*S;
        // error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            let term = 2 * self.mult_terms[i] + 2 * S;
            let error = inp.y * S - term * self.denom_inverses[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }

    fn check_mult(self, inp: Input) {
        // Check y * t
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - inp.y * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for TangentWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for tan(x)
    }

    fn check_ratio(self, inp: Input) {
        // // Check that ratio * y ~= 2
        // let error = 2*S_sq - self.ratio_term * inp.y + 512*S;
        // error.assert_max_bit_size::<LOG_S + 10>();

        // Modified to check y^2
        let error = self.y_sq * S - inp.y * inp.y + 1*S;
        error.assert_max_bit_size::<LOG_S + 2>();
    }

    fn check_inverse(self, inp: Input) {
        // Check if inverse * (mult_term + 4) = 2y
        for i in 0..N_POINTS {
            let term = self.mult_terms[i] + 4 * S;
            let error = 2 * inp.y * S - term * self.denom_inverses[i] + 8*S;
            error.assert_max_bit_size::<LOG_S + 4>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();
    }

    fn check_mult(self, _inp: Input) {
        // Check multiplicative terms = y^2 * GL_ROOT
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - GL_ROOTS_tan[i] * self.y_sq + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for CosineWitness {
    fn check_y_bound(self, inp: Input) {
        // Check that y is at most 1
        inp.y.assert_max_bit_size::<LOG_S>();
    }

    fn check_ratio(self, inp: Input) {
        // Check ratio of tan(x)
        self.tan_witness.check_ratio(Input {
            x: inp.x,
            y: self.tan_x,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: 0,
        });
    }

    fn check_inverse(self, inp: Input) {
        // Check inverse of tan(x)
        self.tan_witness.check_inverse(Input {
            x: inp.x,
            y: self.tan_x,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: 0,
        });

        // Check sqrt_1_p_tan2 * self.y ~= 1
        let error = self.sqrt_1_p_tan2 * inp.y - S_sq + 512*S;
        error.assert_max_bit_size::<LOG_S + 10>();
    }

    fn check_sum(self, inp: Input) {
        // Check sum of tan(x)
        self.tan_witness.check_sum(Input {
            x: inp.x,
            y: self.tan_x,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: 0,
        });
    }

    fn check_mult(self, inp: Input) {
        // Check mult of tan(x)
        self.tan_witness.check_mult(Input {
            x: inp.x,
            y: self.tan_x,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: 0,
        });

        // Check that sqrt_1_p_tan2 * sqrt_1_p_tan2 ~= 1 + wit.tan_x^2
        let error = self.sqrt_1_p_tan2 * self.sqrt_1_p_tan2 - (S_sq + self.tan_x * self.tan_x) + 512*S;
        error.assert_max_bit_size::<LOG_S + 10>();
    }
}

impl GLFunction for PowerWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for power function
    }

    fn check_ratio(self, _inp: Input) {
        // // Check that ratio * (1 - y) ~= (1 + y)
        // let error = (S + inp.y) * S - self.ratio_term_1 * (S - inp.y) + 10000*S;
        // error.assert_max_bit_size::<LOG_S + 15>();

        // // Check that ratio * (1 - x) ~= (1 + x)
        // let error = (S + inp.x) * S - self.ratio_term_2 * (S - inp.x) + 10000*S;
        // error.assert_max_bit_size::<LOG_S + 15>();
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses for first ratio
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] - self.mult_terms_1[i] + S + inp.y;
            let error = (S - inp.y) * S - term * self.denom_inverses_1[i] + 16*S;
            error.assert_max_bit_size::<LOG_S + 5>();
        }
        // Check inverses for second ratio
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] - self.mult_terms_2[i] + S + inp.x;
            let error = (S - inp.x) * inp.k - term * self.denom_inverses_2[i] + 16*S;
            error.assert_max_bit_size::<LOG_S + 5>();
        }
    }

    fn check_sum(self, _inp: Input) {
        // Check the sum
        let lhs = 0;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * (self.denom_inverses_1[i] - self.denom_inverses_2[i]);
        }
        let error = lhs - rhs + 512*S;
        error.assert_max_bit_size::<LOG_S + 10>();
    }

    fn check_mult(self, _inp: Input) {
        // Check y * t for first ratio
        for i in 0..N_POINTS {
            let error = self.mult_terms_1[i] * S - _inp.y * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
        // Check y * t for second ratio
        for i in 0..N_POINTS {
            let error = self.mult_terms_2[i] * S - _inp.x * GL_ROOTS[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }
}

impl GLFunction for SoftmaxWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for softmax
    }

    fn check_ratio(self, inp: Input) {
        // Exp ratio checks
        for i in 0..N_SOFTMAX {
            self.exp_witnesses[i].check_ratio(Input {
                x: inp.vec_x[i],
                y: self.exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }
    }

    fn check_inverse(self, inp: Input) {
        // Exp inverse checks
        for i in 0..N_SOFTMAX {
            self.exp_witnesses[i].check_inverse(Input {
                x: inp.vec_x[i],
                y: self.exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }
    }

    fn check_sum(self, inp: Input) {
        // Exp sum checks
        for i in 0..N_SOFTMAX {
            self.exp_witnesses[i].check_sum(Input {
                x: inp.vec_x[i],
                y: self.exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }

        // Softmax sum check: sum y_i = 1
        let mut sum_y = 0;
        for i in 0..N_SOFTMAX {
            sum_y += inp.vec_y[i];
        }
        let error = sum_y - S + 16;
        error.assert_max_bit_size::<5>();

        // Check scaling: y_i ~= exp_output_i * denom_inverse
        for i in 0..N_SOFTMAX {
            self.check_mult(Input {
                x: inp.vec_x[i],
                y: inp.vec_y[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }
    }

    fn check_mult(self, inp: Input) {
        // Check multiplicative scaling
        for i in 0..N_SOFTMAX {
            let error = inp.vec_y[i] * S - self.exp_outputs[i] * self.denom_inverse + 128*S;
            error.assert_max_bit_size::<LOG_S + 8>();
        }
    }
}