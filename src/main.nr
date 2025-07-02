// Numbers between [-64,64] are quantized to 32 bit integers
// 64 is represented as 2^32
// Scale = 2^(32 - 6) = 2^26
global ONE: Field = 67108864;
global _7_by_90: Field = 5219578;
global _64_by_45: Field = 95443718;
global _4_by_15: Field = 17895697;
global ONE_sq: Field = 67108864 * 67108864;

struct NC5_witness {
    y: Field,
    _y_inv: Field,
    _1_minus_y_inv: Field,
    _y_plus_1_inv: Field,
    _y_plus_3_inv: Field,
    _3y_plus_1_inv: Field,
}

/// Computes inverse of a 32 bit quantized number
unconstrained fn comp_inv(x: Field) -> Field {
    let x_q = x as u128;
    let one_q = 1u128 << 2*26;
    let inv_q = (one_q / x_q) as u128;

    inv_q as Field
}

/// Computes exp(-x) for a 32 bit quantized number using series expansion
unconstrained fn comp_exp(x: Field) -> Field {
    let x_q = x as i64;
    let mut result = ONE as i64;
    let mut term: i64 = ONE as i64;
    
    for i in 1..=25 {
        term = (term * x_q) / ((i as i64) * (ONE as i64));

        if i % 2 == 1 {
            result -= term;
        } else {
            result += term;
        }
    }
    result as Field
}

/// Check that exp(-x) is correct
/// 1. Check smallness of all witness values
/// 2. Check inverses
/// 3. Check NC integration sum
fn nc_5(x: Field, wit: NC5_witness) {
    // Check that y is at most 1
    wit.y.assert_max_bit_size::<26>();

    // Check inverses
    let _1_minus_y = ONE - wit.y;
    let _y_plus_1 = wit.y + ONE;
    let _y_plus_3 = wit.y + 3 * ONE;
    let _3y_plus_1 = 3 * wit.y + ONE;

    (ONE_sq - _1_minus_y * wit._1_minus_y_inv).assert_max_bit_size::<32>();
    (ONE_sq - wit.y * wit._y_inv).assert_max_bit_size::<32>();
    (ONE_sq - _y_plus_1 * wit._y_plus_1_inv).assert_max_bit_size::<32>();
    (ONE_sq - _y_plus_3 * wit._y_plus_3_inv).assert_max_bit_size::<32>();
    (ONE_sq - _3y_plus_1 * wit._3y_plus_1_inv).assert_max_bit_size::<32>();

    // Check NC integration 
    let lhs = x * wit._1_minus_y_inv;
    let rhs = (
        _7_by_90 * wit._y_inv +
        _64_by_45 * wit._y_plus_3_inv +
        _4_by_15 * wit._y_plus_1_inv +
        _64_by_45 * wit._3y_plus_1_inv +
        _7_by_90 * ONE
    );
    
    // Check that lhs is close to rhs 
    let less = lhs.lt(rhs);
    let diff = if less { rhs - lhs } else { lhs - rhs };
    diff.assert_max_bit_size::<40>();
    
}

#[test]
fn test_inv() {
    let x: Field = 14 * ONE + 65756;

    // Safety: Inverse is checked by a constraint
    unsafe{
        let x_inv = comp_inv(x);
        let error = ONE_sq - x * x_inv;
        error.assert_max_bit_size::<32>();
    }
}

#[test]
fn test_exp() {
    let x: Field = 35554432;
    let mut wit = NC5_witness {
        y: 0,
        _y_inv: 0,
        _1_minus_y_inv: 0,
        _y_plus_1_inv: 0,
        _y_plus_3_inv: 0,
        _3y_plus_1_inv: 0
    };

    // Safety: Exponentiation is checked by Newton-Cotes integration
    unsafe{
        let y = comp_exp(x);

        wit = NC5_witness {
            y,
            _y_inv: comp_inv(y),
            _1_minus_y_inv: comp_inv(ONE - y),
            _y_plus_1_inv: comp_inv(y + ONE),
            _y_plus_3_inv: comp_inv(y + 3 * ONE),
            _3y_plus_1_inv: comp_inv(3 * y + ONE)
        };
    }

    nc_5(x, wit);    
}

fn main(x: pub Field) -> pub Field {
    let mut wit = NC5_witness {
        y: 0,
        _y_inv: 0,
        _1_minus_y_inv: 0,
        _y_plus_1_inv: 0,
        _y_plus_3_inv: 0,
        _3y_plus_1_inv: 0
    };

    // Safety: Exponentiation is checked by Newton-Cotes integration
    unsafe{
        let y = comp_exp(x);

        wit = NC5_witness {
            y,
            _y_inv: comp_inv(y),
            _1_minus_y_inv: comp_inv(ONE - y),
            _y_plus_1_inv: comp_inv(y + ONE),
            _y_plus_3_inv: comp_inv(y + 3 * ONE),
            _3y_plus_1_inv: comp_inv(3 * y + ONE)
        };
    }

    nc_5(x, wit);

    wit.y
}