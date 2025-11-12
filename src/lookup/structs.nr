use super::constants::{LOG_BASE, BASE_POWERS, LOG_S, S, MULT_OPS, NUM_TABLES, NUM_SOFTMAX, EXP_TABLES, CUSTOM_TABLE, COEFF_X3, SQRT_2_PI};

pub trait LookupFunction {
    fn check_lookup(self, inp: Input);
}

pub struct LookupWitness <T> {
    pub inp_struct: Input,
    pub wit_struct: T,
}

pub struct Input {
    pub x: Field, // Input value
    pub y: Field, // Output value

    pub vec_x: [Field; NUM_SOFTMAX], // For softmax
    pub vec_y: [Field; NUM_SOFTMAX], // For softmax

    pub k: Field, // For power function
}

// All function types
pub struct InvExpLookupWitness {
    // Witness
    x_decomp: [Field; NUM_TABLES], // Decomposed input value
    lookup_mults: [Field; 2 * NUM_TABLES - 1], // Intermediate products for quantized multiplication
}

pub struct SoftMaxLookupWitness {
    // Witnesses
    exp_wits: [LookupWitness<InvExpLookupWitness>; NUM_SOFTMAX],
    shift: Field,        // Log of denominator
}

pub struct CosineLookupWitness {
    // Witness
    _dummy: Field,
}

pub struct TanLookupWitness {
    // Witness
    _dummy: Field,
}

pub struct TanhLookupWitness {
    // Witness
    exp_output: Field,
    out_sq: Field,
    inv_exp_witness: InvExpLookupWitness,
}

pub struct SigmoidLookupWitness {
    // Witness
    exp_output: Field,
    inv_exp_witness: InvExpLookupWitness,
}

pub struct GeluLookupWitness {
    // Witness
    tanh_lookup_witness: TanhLookupWitness,
    x_sq: Field,
    x_scaled: Field,
    term_2: Field,
    tanh_input: Field,
    tanh_output: Field,
}

pub struct PowerLookupWitness {
    // Witness
    log_x: Field,
    k_log_x: Field,
    inv_exp_witness_x: InvExpLookupWitness,
    inv_exp_witness_y: InvExpLookupWitness,
}

pub struct ErfLookupWitness {
    // Witness
    _dummy: Field,
}

impl LookupFunction for InvExpLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_inv_exp_lookup(inp, self);
    }
}

impl LookupFunction for SoftMaxLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_softmax_lookup(inp, self);
    }
}

impl LookupFunction for CosineLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_cosine_lookup(inp, self);
    }
}

impl LookupFunction for TanLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_tan_lookup(inp, self);
    }
}

impl LookupFunction for TanhLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_tanh_lookup(inp, self);
    }
}

impl LookupFunction for SigmoidLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_sigmoid_lookup(inp, self);
    }
}

impl LookupFunction for GeluLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_gelu_lookup(inp, self);
    }
}

impl LookupFunction for PowerLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_power_lookup(inp, self);
    }
}

impl LookupFunction for ErfLookupWitness {
    fn check_lookup(self, inp: Input) {
        check_erf_lookup(inp, self);
    }
}


fn check_decomp(x: Field, decomp: [Field; NUM_TABLES]) {
    // Check that the decomposition reconstructs the original value
    let mut reconstructed = 0;
    for i in 0..NUM_TABLES {
        reconstructed += decomp[i] * BASE_POWERS[i];
    }
    assert(reconstructed == x);

    // Check that all the decomposed values < BASE
    for i in 0..NUM_TABLES {
        decomp[i].assert_max_bit_size::<LOG_BASE>();
    }
}

/// Multiplies two quantized values and checks that
/// the quantized product is close to the expected value.
fn check_quantized_mult(a_q: Field, b_q: Field, ab_q: Field) {
    // Multiply and keep the top half
    let error = (a_q * b_q - ab_q * S + 16 * S);
    error.assert_max_bit_size::<LOG_S + 5>();
}

/// Checks all the intermediate products
fn check_products(lookup_mults: [Field; 2 * NUM_TABLES - 1]) {
    // Check each multiplication in mult_ops
    for i in 0..MULT_OPS.len() {
        let (a_idx, b_idx, ab_idx) = MULT_OPS[i];
        check_quantized_mult(
            lookup_mults[a_idx as u32],
            lookup_mults[b_idx as u32],
            lookup_mults[ab_idx as u32],
        );
    }
}

pub fn check_inv_exp_lookup(inp: Input, wit: InvExpLookupWitness) {
    // Check validity of the decomposition
    check_decomp(inp.x, wit.x_decomp);

    // Lookup check
    for i in 0..NUM_TABLES {
        // Check that the first NUM_TABLES elements in lookup_mults are the lookup values
        assert(wit.lookup_mults[i] == EXP_TABLES[i][wit.x_decomp[i] as u32]);
    }

    // Multiplication checks
    check_products(wit.lookup_mults);

    // Output range check
    let error = (inp.y - wit.lookup_mults[2 * NUM_TABLES - 2] + 16);
    error.assert_max_bit_size::<5>();
}

pub fn check_tanh_lookup(inp: Input, wit: TanhLookupWitness) {
    // Check that exp_output == exp(-x)
    check_inv_exp_lookup(
        Input {
            x: inp.x,
            y: wit.exp_output,
            vec_x: [0; NUM_SOFTMAX],
            vec_y: [0; NUM_SOFTMAX],
            k: 0,
        },
        wit.inv_exp_witness,
    );

    // Check that out_sq == exp_output^2
    let error = wit.out_sq * S - wit.exp_output * wit.exp_output + 8 * S;
    println(error);
    error.assert_max_bit_size::<LOG_S + 4>();

    // Check tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    let error = inp.y * (S + wit.out_sq) - (S - wit.out_sq) * S + 8 * S;
    error.assert_max_bit_size::<LOG_S + 4>();
}

pub fn check_sigmoid_lookup(inp: Input, wit: SigmoidLookupWitness) {
    // Check that exp_output == exp(-x)
    check_inv_exp_lookup(
        Input {
            x: inp.x,
            y: wit.exp_output,
            vec_x: [0; NUM_SOFTMAX],
            vec_y: [0; NUM_SOFTMAX],
            k: 0,
        },
        wit.inv_exp_witness,
    );

    // Check that denom_inverse is the inverse of (1 + exp_output)
    let error = inp.y * (S + wit.exp_output) - S * S + 16 * S;
    println(error);
    error.assert_max_bit_size::<LOG_S + 5>();
}

pub fn check_gelu_lookup(inp: Input, wit: GeluLookupWitness) {
    // Check that tanh_output == tanh(tanh_input)
    check_tanh_lookup(
        Input {
            x: wit.tanh_input,
            y: wit.tanh_output,
            vec_x: [0; NUM_SOFTMAX],
            vec_y: [0; NUM_SOFTMAX],
            k: 0,
        },
        wit.tanh_lookup_witness,
    );

    // Check x_sq == x * x
    let error = wit.x_sq * S - inp.x * inp.x + 8 * S;
    error.assert_max_bit_size::<LOG_S + 4>();

    // Check x_scaled == 0.044715 * x
    let error = wit.x_scaled * S - inp.x * COEFF_X3 + 8 * S;
    error.assert_max_bit_size::<LOG_S + 4>();

    // Check term_2 = x_scaled * x_sq
    let error = wit.term_2 * S - wit.x_scaled * wit.x_sq + 2*S;
    error.assert_max_bit_size::<LOG_S + 2>();

    // Check tanh_input = sqrt(2/pi) * (x + term_2)
    let error = (inp.x + wit.term_2) * SQRT_2_PI - wit.tanh_input * S + 4*S;
    error.assert_max_bit_size::<LOG_S + 3>();

    // Check gelu output: 2 * y = x * (1 + tanh_output)
    let error = 2 * inp.y * S - inp.x * (S + wit.tanh_output) + 4*S;
    error.assert_max_bit_size::<LOG_S + 3>();
}

pub fn check_power_lookup(inp: Input, wit: PowerLookupWitness) {
    // Check that x = e^(log_x)
    check_inv_exp_lookup(
        Input {
            x: wit.log_x,
            y: inp.x,
            vec_x: [0; NUM_SOFTMAX],
            vec_y: [0; NUM_SOFTMAX],
            k: 0,
        },
        wit.inv_exp_witness_x,
    );

    // Check that y = e^(k * log_x)
    check_inv_exp_lookup(
        Input {
            x: wit.k_log_x,
            y: inp.y,
            vec_x: [0; NUM_SOFTMAX],
            vec_y: [0; NUM_SOFTMAX],
            k: 0,
        },
        wit.inv_exp_witness_y,
    );

    // Check that k_log_x = k * log_x
    let error = wit.k_log_x * S - inp.k * wit.log_x + 8 * S;
    error.assert_max_bit_size::<LOG_S + 4>();
}

pub fn check_erf_lookup(inp: Input, _wit: ErfLookupWitness) {
    assert(inp.y == CUSTOM_TABLE[inp.x as u32]);
}

pub fn check_softmax_lookup(inp: Input, wit: SoftMaxLookupWitness) {
    // Check that exp_wits[i].x == shift - vec_x[i]
    for i in 0..NUM_SOFTMAX {
        assert(wit.exp_wits[i].inp_struct.x == wit.shift - inp.vec_x[i]);
    }

    // Check each exp lookup
    for i in 0..NUM_SOFTMAX {
        check_inv_exp_lookup(wit.exp_wits[i].inp_struct, wit.exp_wits[i].wit_struct);
    }

    // Check sum ~= 1
    let mut sum_exp = 0;
    for i in 0..NUM_SOFTMAX {
        sum_exp += wit.exp_wits[i].inp_struct.y;
    }
    let error = (sum_exp - S + 8);
    error.assert_max_bit_size::<4>();

    // Check vec_y[i] ~= exp(vec_x[i])
    for i in 0..NUM_SOFTMAX {
        let error = (inp.vec_y[i] - wit.exp_wits[i].inp_struct.y);
        error.assert_max_bit_size::<4>();
    }
}

pub fn check_cosine_lookup(inp: Input, _wit: CosineLookupWitness) {
    // One lookup call to a cosine table
    // Cannot decompose cosine unless using a half angle formula like cos(2x) = 2cos^2(x) - 1
    // but this only reduces the table size by a factor of 2
    assert(inp.y == CUSTOM_TABLE[inp.x as u32]);
}

pub fn check_tan_lookup(inp: Input, _wit: TanLookupWitness) {
    // One lookup call to a tangent table
    assert(inp.y == CUSTOM_TABLE[inp.x as u32]);
}