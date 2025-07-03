// Approximation for exp(-x) using Newton-Cotes integration
// Scaling factor = 2^100
global S: Field = 1267650600228229401496703205376;
global S_sq : Field = S * S; // 2^200


// Newton-Cotes integration order -- has to be odd
global NC_ORDER: u32 = 19;

// Statement and witness
struct NC_witness {
    // Statement
    pub x: Field, // Input value
    pub signs: [Field; NC_ORDER], // Signs for Newton-Cotes integration
    pub weights: [Field; NC_ORDER], // Weights for Newton-Cotes integration

    // Witness
    y: Field, // exp(-x)
    lhs_inverse: Field, // Inverse of (1 - y)
    rhs_inverses: [Field; NC_ORDER], // Inverses of (i + (n-i) y)
}

fn check_inverse(x: Field, x_inv: Field) {
    // Check that x * x_inv ~= 1
    let error = S_sq - x * x_inv + 10*S;    
    error.assert_max_bit_size::<120>();
}

fn check_nc_int(wit: NC_witness) {
    // Check that y is at most 1
    wit.y.assert_max_bit_size::<100>();

    // Check inverses
    // Optimization: We can skip one of the checks
    check_inverse(S - wit.y, wit.lhs_inverse);
    for i in 0..NC_ORDER {
        let term = Field::from(i as u32) * S + Field::from(NC_ORDER as u32 - 1 - i) * wit.y;
        check_inverse(term, wit.rhs_inverses[i]);
    }

    // Check the sum
    let lhs = wit.x * wit.lhs_inverse;
    let mut rhs = 0;
    for i in 0..NC_ORDER {
        rhs += wit.signs[i] * wit.weights[i] * wit.rhs_inverses[i];
    }
    let less = lhs.lt(rhs);
    let diff = if less { rhs - lhs } else { lhs - rhs };

    diff.assert_max_bit_size::<180>();       
}

fn main(wit: NC_witness) {
    check_nc_int(wit);
}