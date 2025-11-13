use super::constants::{LOG_S, S, S_sq, N_POINTS, N_SOFTMAX, GL_ROOTS, GL_ROOTS_tan, GL_WEIGHTS, COEFF_X3, SQRT_2_PI, SQRT_PI, TWO_PI, PI};

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

// y = gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub struct GeluWitness {
    pub tanh_witness: TanhWitness,
    pub tanh_input: Field, // sqrt(2/pi) * (x + 0.044715 * x^3)
    pub tanh_output: Field, // tanh(tanh_input)
    
    pub x_scaled: Field, // 0.044715 * x
    pub x_sq: Field, // x^2
    pub term_2: Field, // x_scaled * x_sq
}

// y = erf(x)
pub struct ErfWitness {
    pub inv_exp_witnesses: [InverseExponentialWitness; N_POINTS],
    pub inv_exp_inputs: [Field; N_POINTS], // x^2 / 4 * GL_ROOTS_tan[i]
    pub inv_exp_outputs: [Field; N_POINTS],

    pub erf_sum: Field, // rescaled quad sum check
    pub x_sq_by_4: Field, // x^2 / 4
}

// Macro Benchmark -- Kepler's equation
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
    pub cos_witness: CosineWitness,
}

// Macro Benchmark -- Bancor's formula
pub struct BancorWitness {
    pub S_supply: Field,
    pub R_reserve: Field,
    pub E_payment: Field,
    pub F_reserve_ratio: Field,
    pub T_tokens: Field,

    pub pow_input: Field,
    pub pow_output: Field,
    pub pow_wit: PowerWitness,
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
        let error = lhs - rhs + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();
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
        let error = self.y_sq * S - inp.y * inp.y + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }

    fn check_inverse(self, inp: Input) {
        // Check if inverse * (mult_term + 4) = 2y
        for i in 0..N_POINTS {
            let term = self.mult_terms[i] + 4 * S;
            let error = 2 * inp.y * S - term * self.denom_inverses[i] + 16*S;
            error.assert_max_bit_size::<LOG_S + 5>();
        }
    }

    fn check_sum(self, inp: Input) {
        // Check the sum
        let lhs = inp.x * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.denom_inverses[i];
        }
        let error = lhs - rhs + 32*S;
        error.assert_max_bit_size::<LOG_S + 6>();
    }

    fn check_mult(self, _inp: Input) {
        // Check multiplicative terms = y^2 * GL_ROOT
        // Error bounds increase for kepler use case
        for i in 0..N_POINTS {
            let error = self.mult_terms[i] * S - GL_ROOTS_tan[i] * self.y_sq + 16*S;
            error.assert_max_bit_size::<LOG_S + 5>();
        }
    }
}

impl GLFunction for CosineWitness {
    fn check_y_bound(self, inp: Input) {
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
        let error = self.sqrt_1_p_tan2 * inp.y - S_sq + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();
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
        let error = self.sqrt_1_p_tan2 * self.sqrt_1_p_tan2 - (S_sq + self.tan_x * self.tan_x) + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();
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
            let error = (S - inp.y) * S - term * self.denom_inverses_1[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
        // Check inverses for second ratio
        for i in 0..N_POINTS {
            let term = GL_ROOTS[i] - self.mult_terms_2[i] + S + inp.x;
            let error = (S - inp.x) * inp.k - term * self.denom_inverses_2[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }
    }

    fn check_sum(self, _inp: Input) {
        // Check the sum
        let lhs = 0;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * (self.denom_inverses_1[i] - self.denom_inverses_2[i]);
        }
        let error = lhs - rhs + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();
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

impl GLFunction for GeluWitness {
    fn check_y_bound(self, _inp: Input) {
        // No need to check y bound for gelu
    }

    fn check_ratio(self, _inp: Input) {
        // No ratio check for gelu
    }

    fn check_inverse(self, _inp: Input) {
        // Check tanh inverse
        self.tanh_witness.check_inverse(Input {
            x: self.tanh_input,
            y: self.tanh_output,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_sum(self, _inp: Input) {
        // Check tanh sum
        self.tanh_witness.check_sum(Input {
            x: self.tanh_input,
            y: self.tanh_output,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_mult(self, inp: Input) {
        // Check tanh mult
        self.tanh_witness.check_mult(Input {
            x: self.tanh_input,
            y: self.tanh_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: 0,
        });

        // Check x_sq = x * x
        let error = self.x_sq * S - inp.x * inp.x + 2*S;
        error.assert_max_bit_size::<LOG_S + 2>();

        // Check x_scaled = 0.044715 * x
        let error = self.x_scaled * S - inp.x * COEFF_X3 + 2*S;
        println(error);
        error.assert_max_bit_size::<LOG_S + 2>();

        // Check term_2 = x_scaled * x_sq 
        let error = self.term_2 * S - self.x_scaled * self.x_sq + 2*S;
        error.assert_max_bit_size::<LOG_S + 2>();

        // Check tanh_input = sqrt(2/pi) * (x + term_2)
        let error = (inp.x + self.term_2) * SQRT_2_PI - self.tanh_input * S + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();

        // Check gelu output: 2 * y = x * (1 + tanh_output)
        let error = 2 * inp.y * S - inp.x * (S + self.tanh_output) + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();
    }
}

impl GLFunction for ErfWitness {
    fn check_y_bound(self, inp: Input) {
        inp.y.assert_max_bit_size::<LOG_S>();
        for i in 0..N_POINTS {
            println(self.inv_exp_outputs[i]);
            self.inv_exp_outputs[i].assert_max_bit_size::<LOG_S + 1>();
        }
    }

    fn check_ratio(self, _inp: Input) {
        // No ratio check for erf
    }

    fn check_inverse(self, inp: Input) {
        // Check inverses
        for i in 0..N_POINTS {
            self.inv_exp_witnesses[i].check_inverse(Input {
                x: self.inv_exp_inputs[i],
                y: self.inv_exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }
    }

    fn check_sum(self, inp: Input) {
        for i in 0..N_POINTS {
            self.inv_exp_witnesses[i].check_sum(Input {
                x: self.inv_exp_inputs[i],
                y: self.inv_exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }

        // Check sum for erf
        let lhs = self.erf_sum * S;
        let mut rhs = 0;
        for i in 0..N_POINTS {
            rhs += GL_WEIGHTS[i] * self.inv_exp_outputs[i];
        }
        let error = lhs - rhs + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();

    }

    fn check_mult(self, inp: Input) {
        for i in 0..N_POINTS {
            self.inv_exp_witnesses[i].check_mult(Input {
                x: self.inv_exp_inputs[i],
                y: self.inv_exp_outputs[i],
                vec_x: inp.vec_x,
                vec_y: inp.vec_y,
                k: 0,
            });
        }
        
        // Check x_sq_by_4 = x^2 / 4
        let error = 4 * self.x_sq_by_4 * S - inp.x * inp.x + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();

        // Check inv_exp_inputs = x_sq_by_4 * GL_ROOTS_tan[i]
        for i in 0..N_POINTS {
            let error = self.inv_exp_inputs[i] * S - self.x_sq_by_4 * GL_ROOTS_tan[i] + 4*S;
            error.assert_max_bit_size::<LOG_S + 3>();
        }

        // Check erf output
        let error = inp.y * SQRT_PI - self.erf_sum * inp.x + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();
    }

}

impl GLFunction for KeplerWitness {
    fn check_y_bound(self, _inp: Input) {
        self.cos_witness.check_y_bound(Input {
            x: self.E - self.selector * PI,
            y: self.cos_E_shifted,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_ratio(self, _inp: Input) {
        self.cos_witness.check_ratio(Input {
            x: self.E - self.selector * PI,
            y: self.cos_E_shifted,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_inverse(self, _inp: Input) {
        self.cos_witness.check_inverse(Input {
            x: self.E - self.selector * PI,
            y: self.cos_E_shifted,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_sum(self, _inp: Input) {
        self.cos_witness.check_sum(Input {
            x: self.E - self.selector * PI,
            y: self.cos_E_shifted,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });
    }

    fn check_mult(self, _inp: Input) {
        self.cos_witness.check_mult(Input {
            x: self.E - self.selector * PI,
            y: self.cos_E_shifted,
            vec_x: _inp.vec_x,
            vec_y: _inp.vec_y,
            k: 0,
        });

        // Check shifted cosine
        assert(self.selector * (self.selector - 1) * (self.selector - 2) == 0);
        let coeff = 2 * self.selector * self.selector - 4 * self.selector + 1;
        let error = self.cos_E_shifted * coeff - self.cos_E + 1;
        error.assert_max_bit_size::<1>();

        // Check a_sq = a * a
        let error = self.a_sq * S - self.a * self.a + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();

        // Check a^3 = P_orbit^2
        let error = self.P_orbit * self.P_orbit - self.a * self.a_sq + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();

        // Check b = a * sqrt(1 - e^2)
        let error = self.b * S - self.a * self.sqrt_1_m_e2 + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();

        // Check M = 2pi * dt / P_orbit
        let error = self.M * self.P_orbit - TWO_PI * self.dt + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();

        // Check M = E - e * sin(E)
        let error = self.M * S - self.E * S + self.e * self.sin_E + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();

        // Check sin_E^2 + cos_E^2 = 1
        let error = self.sin_E * self.sin_E + self.cos_E * self.cos_E - S_sq + 16*S;
        error.assert_max_bit_size::<LOG_S + 5>();

        // Check sqrt_1_m_e2 = sqrt(1 - e^2)
        let error = self.sqrt_1_m_e2 * self.sqrt_1_m_e2 + self.e * self.e - S_sq + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();

        // Check x = a * (cos_E - e)
        let error = self.x * S - self.a * (self.cos_E - self.e) + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();

        // Check y = b * sin_E
        let error = self.y * S - self.b * self.sin_E + 4*S;
        error.assert_max_bit_size::<LOG_S + 3>();

    }
}

impl GLFunction for BancorWitness {
    fn check_y_bound(self, inp: Input) {
        self.pow_wit.check_y_bound(Input {
            x: self.pow_input,
            y: self.pow_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: inp.k,
        });
    }

    fn check_ratio(self, inp: Input) {
        self.pow_wit.check_ratio(Input {
            x: self.pow_input,
            y: self.pow_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: inp.k,
        });
    }

    fn check_inverse(self, inp: Input) {
        self.pow_wit.check_inverse(Input {
            x: self.pow_input,
            y: self.pow_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: inp.k,
        });
    }

    fn check_sum(self, inp: Input) {
        self.pow_wit.check_sum(Input {
            x: self.pow_input,
            y: self.pow_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: inp.k,
        });
    }

    fn check_mult(self, inp: Input) {
        self.pow_wit.check_mult(Input {
            x: self.pow_input,
            y: self.pow_output,
            vec_x: inp.vec_x,
            vec_y: inp.vec_y,
            k: inp.k,
        });

        // Check pow_input = (R + E) / R
        let error = self.pow_input * self.R_reserve - (self.E_payment + self.R_reserve) * S + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();

        // Check T_tokens = S_supply * (pow_output - 1)
        let error = self.T_tokens * S - self.S_supply * (self.pow_output - S) + 8*S;
        error.assert_max_bit_size::<LOG_S + 4>();

    }
}