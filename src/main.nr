// Approximation for exp(-x) using Newton-Cotes integration
mod constants;
use constants::{S_sq, S, NC_ORDER, signs, weights};

// Statement and witness
struct NC_witness {
    // Statement
    pub x: Field, // Input value
    pub y: Field, // Output = exp(-x)

    // Witness
    lhs_inverse: Field, // Inverse of (1 - y)
    rhs_inverses: [Field; NC_ORDER], // Inverses of (i + (n-i) y)
}

fn check_inverse(x: Field, x_inv: Field) {
    // Check that x * x_inv ~= 1
    let error = S_sq - x * x_inv + 100*S;
    error.assert_max_bit_size::<110>();
}

fn check_nc_int(wit: NC_witness) {
    // Check that y is at most 1
    wit.y.assert_max_bit_size::<100>();

    // Check inverses
    // Optimization: We can skip one of the checks
    check_inverse(S - wit.y, wit.lhs_inverse);
    for i in 0..NC_ORDER {
        let term = (i as Field) * S + ((NC_ORDER - 1 - i) as Field) * wit.y;
        check_inverse(term, wit.rhs_inverses[i]);
    }

    // Check the sum
    let lhs = wit.x * wit.lhs_inverse;
    let mut rhs = 0;
    for i in 0..NC_ORDER {
        rhs += signs[i] * weights[i] * wit.rhs_inverses[i];
    }
    let less = lhs.lt(rhs);
    let diff = if less { rhs - lhs } else { lhs - rhs };

    diff.assert_max_bit_size::<180>();       
}

fn main(wit: NC_witness) {
    check_nc_int(wit);
}